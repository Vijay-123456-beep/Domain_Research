import cv2
import numpy as np
import re
import os
import json
import easyocr

from figure_classifier import FigureClassifier
from subplot_splitter import SubplotSplitter
from axis_detector import AxisDetector
from tick_detector import TickDetector
from curve_segmenter import CurveSegmenter
from marker_detector import MarkerDetector
from legend_matcher import LegendMatcher
from utils.ocr import OCRWorker
from curve_tracer import CurveTracer
from scale_calibrator import ScaleCalibrator
from curve_smoother import CurveSmoother
try:
    from models_v2 import YOLOEdgeDetector
except ImportError:
    YOLOEdgeDetector = None

# Pre-compiled regex patterns for performance
TICK_PATTERN = re.compile(r'^[\d\.\-]+$')
AXIS_KEYWORDS = {
    'y': ['capacitance', 'capacity', 'current', 'voltage', 'potential', 
          'energy', 'power', 'density', 'magnetization', 'intensity'],
    'x': ['cycle', 'time', 'scan rate', 'current density', 
          'voltage', 'potential', 'frequency', 'field']
}

class GraphDigitizer:
    """
    Orchestrator for scientific plot digitization.
    """
    def __init__(self, reader=None):
        self.reader = reader
        if not self.reader:
            try:
                # Initialize OCR reader once
                self.reader = easyocr.Reader(['en'])
            except Exception as e:
                print(f"Warning: EasyOCR initialization failed: {e}")
                self.reader = None
        self.classifier = FigureClassifier(reader)
        self.splitter = SubplotSplitter(reader)
        self.axis_detector = AxisDetector()
        self.tick_detector = TickDetector()
        self.segmenter = CurveSegmenter()
        self.tracer = CurveTracer()
        self.calibrator = ScaleCalibrator()
        self.smoother = CurveSmoother(smoothing_factor=0.5)
        self.legend_matcher = LegendMatcher(reader)
        self.marker_detector = MarkerDetector()
        self.ocr = OCRWorker(reader)
        self.ai_edge_detector = YOLOEdgeDetector() if YOLOEdgeDetector else None

    def classify_figure(self, img_np):
        return self.classifier.classify(img_np)

    def digitize(self, img_np):
        """
        Processes a plot image and extracts structured datasets.
        """
        # Validate image first
        if img_np is None or img_np.size == 0:
            return {"error": "Empty or invalid image", "series": []}
        if len(img_np.shape) < 2:
            return {"error": "Invalid image dimensions", "series": []}
        if img_np.shape[0] < 50 or img_np.shape[1] < 50:
            return {"error": f"Image too small: {img_np.shape}", "series": []}
        
        # 1. Classification
        classification = self.classify_figure(img_np)
        if classification["action"] == "skip_extraction":
            return {"error": f"Skipping non-graph figure ({classification['figure_type']})", "classification": classification, "series": []}

        # 2. Subplot Splitting
        subplots = self.splitter.split(img_np)
        results = []

        for panel in subplots:
            panel_img = panel["image"]
            panel_label = panel["label"]
            
            # Validate panel image
            if panel_img is None or panel_img.size == 0:
                continue
            if panel_img.shape[0] < 20 or panel_img.shape[1] < 20:
                continue
            
            # 3. Axis Detection
            axes_info = None
            if self.ai_edge_detector:
                 try:
                      axes_info = self.ai_edge_detector.detect_plot_area(panel_img)
                      print("  [AI Edge Detection] YOLOv8 successfully located plot axes area.")
                 except Exception as e:
                      print(f"  [AI Edge Detection Failed] Falling back to Canny Edge: {e}")
            
            if not axes_info:
                 axes_info = self.axis_detector.detect(panel_img)
                 
            if not axes_info:
                continue
            
            # Validate bbox coordinates — skip panels where axes are inverted/degenerate
            bbox = axes_info.get("bbox", [0, 0, 0, 0])
            x1_b, y1_b, x2_b, y2_b = bbox
            if x2_b <= x1_b or y2_b <= y1_b:
                continue
            # Also ensure the bbox region has a minimum meaningful size
            if (x2_b - x1_b) < 10 or (y2_b - y1_b) < 10:
                continue
            
            # 3.5 Extract Axis Titles using OCR
            axis_titles = self._extract_axis_titles(panel_img, axes_info)
            if not axis_titles.get("x", "").strip() or not axis_titles.get("y", "").strip():
                try:
                    import base64
                    from llm_validator import evaluate_graph_image_llm
                    _, buffer = cv2.imencode('.jpg', panel_img)
                    b64 = base64.b64encode(buffer).decode('utf-8')
                    llm_res = evaluate_graph_image_llm(b64)
                    print(f"  [LLM Rescue] Plot OCR Failed. Vision AI Extracted: X='{llm_res.get('x_axis_label')}' Y='{llm_res.get('y_axis_label')}'")
                    if not axis_titles.get("x", "").strip() and llm_res.get("x_axis_label"):
                        axis_titles["x"] = llm_res.get("x_axis_label")
                    if not axis_titles.get("y", "").strip() and llm_res.get("y_axis_label"):
                        axis_titles["y"] = llm_res.get("y_axis_label")
                except Exception as e:
                    print(f"  [LLM Rescue Failed] {e}")
            
            # 4. Tick & Scale Calibration
            ticks = self.tick_detector.detect(panel_img, axes_info)
            # OCR for tick values (to be implemented in orchestrator for flow)
            tick_values = self._extract_tick_values(panel_img, ticks, axes_info)
            calib_info = self.calibrator.calibrate(axes_info, ticks, tick_values)
            
            # 5. Legend Matching
            # We first segment to get colors
            segmented_curves = self.segmenter.segment(panel_img, axes_info)
            hues = [c["hue"] for c in segmented_curves]
            legend_map = self.legend_matcher.match(panel_img, hues)
            
            # 6. Curve & Marker Extraction
            series_data = []
            for curve in segmented_curves:
                points_px = self.tracer.trace(curve["mask"])
                points_val = self.calibrator.apply_calibration(points_px, calib_info)
                
                # Apply V2 Noise Cleaning & Spline Smoothing
                points_val = self.smoother.clean_and_smooth(points_val)
                
                label = legend_map.get(curve["hue"], f"Series_{len(series_data)}")
                series_data.append({
                    "name": label,
                    "points": points_val
                })
                
            # Add markers
            markers = self.marker_detector.detect(panel_img, axes_info)
            if markers:
                # Group markers by shape and quantized color to handle slight pixel variations
                marker_groups = {}
                for m in markers:
                    color_tup = tuple([int(round(c / 32) * 32) for c in m["color"]])
                    group_key = f"{m['shape']}_{color_tup}"
                    if group_key not in marker_groups:
                        marker_groups[group_key] = {"color": m["color"], "points_px": []}
                    marker_groups[group_key]["points_px"].append(m["pos"])
                
                for g_idx, (g_key, g_data) in enumerate(marker_groups.items()):
                    points_px = g_data["points_px"]
                    # Sort points by X coordinate (typical for scatter plots)
                    points_px.sort(key=lambda p: p[0])
                    
                    points_val = self.calibrator.apply_calibration(points_px, calib_info)
                    
                    # Convert RGB to OpenCV Hue (0-180) to try to match legend
                    r, g, b = g_data["color"]
                    v_max = max(r, g, b)
                    v_min = min(r, g, b)
                    hue = 0
                    if v_max != v_min:
                        if v_max == r:
                            hue = 60 * ((g - b) / (v_max - v_min))
                        elif v_max == g:
                            hue = 60 * ((b - r) / (v_max - v_min) + 2)
                        elif v_max == b:
                            hue = 60 * ((r - g) / (v_max - v_min) + 4)
                        if hue < 0:
                            hue += 360
                    cv_hue = int(hue / 2)
                    
                    label = legend_map.get(cv_hue, f"Scatter_{len(series_data)}")
                    
                    series_data.append({
                        "name": label,
                        "points": points_val
                    })

            subplot_res = {
                "figure": f"Panel {panel_label}",
                "series": series_data,
                "scale": {
                    "y_title": axis_titles.get("y", ""),
                    "x_title": axis_titles.get("x", ""),
                    "y_unit": "",
                    "x_unit": ""
                }
            }
            results.append(subplot_res)

        return results[0] if len(results) == 1 else {"multi_panel": results}

    def _extract_axis_titles(self, img_np, axes_info):
        """
        Extract axis titles (labels) using OCR from regions near the axes.
        Returns dict with 'x' and 'y' titles.
        """
        titles = {"x": "", "y": ""}
        if not self.reader:
            return titles
        
        try:
            h, w = img_np.shape[:2]
            origin_x, origin_y = axes_info.get("origin", (w//2, h//2))
            x_axis_end = axes_info.get("x_axis_end", (w, origin_y))
            y_axis_end = axes_info.get("y_axis_end", (origin_x, 0))
            
            # Define regions for axis title search - BROADER SEARCH AREAS
            # Y-axis title: entire left side of plot area (often rotated)
            # X-axis title: below X-axis, full width of plot area
            
            # Y-axis region: left side, middle vertical area (wider search)
            y_region_x1 = 0
            y_region_y1 = max(0, int(y_axis_end[1]) - 50)  # Above axis end
            y_region_x2 = max(30, int(origin_x) - 20)  # Left of origin
            y_region_y2 = min(h, int(origin_y) + 50)  # Below origin
            
            # X-axis region: below X-axis, full plot width
            x_region_x1 = max(0, int(origin_x) - 50)
            x_region_y1 = min(h - 20, int(origin_y) + 10)  # Just below axis
            x_region_x2 = min(w, int(x_axis_end[0]) + 50)
            x_region_y2 = min(h, int(origin_y) + 120)  # Deeper search below
            
            y_img = img_np[y_region_y1:y_region_y2, y_region_x1:y_region_x2]
            x_img = img_np[x_region_y1:x_region_y2, x_region_x1:x_region_x2]
            
            # Process Y-axis
            if y_img.size > 0:
                y_results = self.reader.readtext(y_img, detail=1, paragraph=False, width_ths=0.8)
                for (bbox, text, prob) in y_results:
                    if prob > 0.2 and len(text.strip()) > 1:
                        if not TICK_PATTERN.match(text.strip()):
                            titles["y"] = text.strip()
                            break
            
            # Process X-axis
            if x_img.size > 0:
                x_results = self.reader.readtext(x_img, detail=1, paragraph=False, width_ths=0.8)
                for (bbox, text, prob) in x_results:
                    if prob > 0.2 and len(text.strip()) > 1:
                        # Reject tick-like labels
                        if not TICK_PATTERN.match(text.strip()):
                            # Handle truncated text like "time (see" -> "time (sec)"
                            text_clean = text.strip().lower()
                            if text_clean.startswith('time (see'):
                                text_clean = 'time (sec)'
                            titles["x"] = text_clean
                            break
            
            # If still no titles found, try searching the full image for common labels
            if not titles["y"] or not titles["x"]:
                full_ocr = self.reader.readtext(img_np, detail=0)
                for text in full_ocr:
                    text_clean = text.strip().lower()
                    if len(text_clean) > 3:
                        # Check for common Y-axis labels
                        if not titles["y"]:
                            for kw in AXIS_KEYWORDS['y']:
                                if kw in text_clean and not TICK_PATTERN.match(text_clean):
                                    titles["y"] = text.strip()
                                    break
                        # Check for common X-axis labels
                        if not titles["x"]:
                            for kw in AXIS_KEYWORDS['x']:
                                if kw in text_clean and not TICK_PATTERN.match(text_clean):
                                    # Handle truncated text like "time (see" -> "time (sec)"
                                    if text_clean.startswith('time (see'):
                                        text_clean = 'time (sec)'
                                    titles["x"] = text.strip()
                                    break
                                
        except Exception as e:
            pass  # Return empty titles on error
        
        return titles

    def _extract_tick_values(self, img_np, ticks, axes_info):
        """
        Internal logic to map OCR'd numbers to specific tick pixels.
        """
        val_map = {"x": {}, "y": {}}
        if not self.reader: return val_map
        
        ocr_res = self.reader.readtext(img_np)
        origin_x, origin_y = axes_info["origin"]
        
        for (bbox, text, prob) in ocr_res:
            num_match = re.search(r"([-+]?\d+(?:\.\d+)?)", text)
            if not num_match: continue
            val = float(num_match.group(1))
            
            # Center of bbox
            cx = (bbox[0][0] + bbox[2][0]) / 2
            cy = (bbox[0][1] + bbox[2][1]) / 2
            
            # Match to nearest tick
            if cy > origin_y: # Potential X label
                for tx in ticks.get("x_ticks", []):
                    if abs(cx - tx) < 15:
                        val_map["x"][tx] = val
                        break
            elif cx < origin_x: # Potential Y label
                for ty in ticks.get("y_ticks", []):
                    if abs(cy - ty) < 15:
                        val_map["y"][ty] = val
                        break
        return val_map

if __name__ == "__main__":
    # Test stub
    pass
