import cv2
import numpy as np
import re

class FigureClassifier:
    """
    Classifies figures based on geometric, structural, and semantic signals.
    """
    def __init__(self, reader=None):
        self.reader = reader # EasyOCR reader

    def classify(self, img_np):
        """
        Classifies the image as 'graph', 'microscopy', or 'schematic'.
        Returns: {figure_type, action, confidence, details}
        """
        if img_np is None or img_np.size == 0:
            return {"figure_type": "unknown", "action": "skip", "confidence": 0, "details": "empty image"}
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # 1. Geometric Check (Axes)
        has_axes = self._check_axes(gray)
        
        # 2. Structural Check (Numeric Ticks)
        has_ticks = self._check_ticks(gray)
        
        # 3. Semantic Check (OCR Keywords)
        ocr_results = []
        if self.reader:
            ocr_results = self.reader.readtext(img_np)
        
        keyword_signal = self._check_keywords(ocr_results)
        
        # Scoring
        score = 0
        if has_axes: score += 40
        if has_ticks: score += 30
        
        if keyword_signal == "graph": score += 30
        elif keyword_signal == "microscopy": score -= 60
        elif keyword_signal == "schematic": score -= 40
        
        confidence = max(0, min(100, score))
        
        figure_type = "graph"
        action = "extract_plot_data"
        
        if confidence < 40:
            action = "skip_extraction"
            figure_type = keyword_signal if keyword_signal != "graph" else "unknown_non_graph"
        
        return {
            "figure_type": figure_type,
            "action": action,
            "confidence": confidence,
            "details": {
                "has_axes": has_axes,
                "has_ticks": has_ticks,
                "keyword_signal": keyword_signal
            }
        }

    def _check_axes(self, gray):
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        
        if lines is None: return False
        
        horizontals = []
        verticals = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y1 - y2) < 2: # Horizontal
                horizontals.append(line[0])
            elif abs(x1 - x2) < 2: # Vertical
                verticals.append(line[0])
        
        return len(horizontals) >= 1 and len(verticals) >= 1

    def _check_ticks(self, gray):
        # Ticks are small perpendicular lines near axes. 
        # For simplicity in classification, we check for high density of small edges near possible axis regions.
        return True # Placeholder for more complex tick structural analysis

    def _check_keywords(self, ocr_results):
        text = " ".join([res[1].lower() for res in ocr_results])
        
        microscopy_keywords = ["tem", "sem", "afm", "scale bar", "magnification", "nanometer", "μm", "nm"]
        schematic_keywords = ["setup", "scheme", "illustration", "proposed", "apparatus", "diagram"]
        graph_keywords = ["capacitance", "voltage", "current", "density", "capacity", "discharge", "cycle", "frequency"]
        
        for kw in microscopy_keywords:
            if kw in text: return "microscopy"
        for kw in schematic_keywords:
            if kw in text: return "schematic"
        for kw in graph_keywords:
            if kw in text: return "graph"
            
        return "graph" # Default to graph if no strong negative keywords
