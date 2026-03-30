import cv2
import numpy as np
try:
    from models_v2 import CurveMaskRCNN
except ImportError:
    CurveMaskRCNN = None

class CurveSegmenter:
    """
    Segments individual curves from the plot using PyTorch Mask R-CNN if available,
    falling back to legacy HSV color clustering.
    """
    def __init__(self):
        self.ai_segmenter = CurveMaskRCNN() if CurveMaskRCNN else None

    def segment(self, img_np, axes_info):
        """
        Returns a list of binary masks, one for each detected curve.
        """
        if not axes_info: return []
        if img_np is None or img_np.size == 0:
            return []
            
        # V2 AI Pipeline: Mask R-CNN Instance Segmentation
        if self.ai_segmenter:
            try:
                # Mask R-CNN handles overlapping curves and grayscale natively
                masks = self.ai_segmenter.segment(img_np, axes_info)
                if masks: 
                    # Reformat from AI output dict to expected legacy format to not break downstream
                    legacy_format = []
                    for m in masks:
                        # Extract primary hue from the color signature tuple
                        r, g, b = m["color_signature"]
                        v_max, v_min = max(r, g, b), min(r, g, b)
                        hue = 0
                        if v_max != v_min:
                            if v_max == r: hue = 60 * ((g - b) / (v_max - v_min))
                            elif v_max == g: hue = 60 * ((b - r) / (v_max - v_min) + 2)
                            elif v_max == b: hue = 60 * ((r - g) / (v_max - v_min) + 4)
                        if hue < 0: hue += 360
                        
                        legacy_format.append({"mask": m["mask"], "hue": int(hue/2)})
                    print(f"  [AI Segmentation] Mask R-CNN successfully isolated {len(legacy_format)} curves.")
                    return legacy_format
            except Exception as e:
                print(f"  [AI Segmentation Failed] Falling back to HSV Heuristics: {e}")
                
        # V1 Legacy Pipeline: HSV Color Clustering Fallback
        x1, y1, x2, y2 = axes_info["bbox"]
        roi = img_np[int(y1):int(y2), int(x1):int(x2)]
        if roi.size == 0:
            return []
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        peaks = self._find_peaks(h_hist)
        
        series_masks = []
        for hue_peak in peaks:
            lower = np.array([max(0, hue_peak-10), 50, 50])
            upper = np.array([min(179, hue_peak+10), 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
            
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            if cv2.countNonZero(mask) > 100:
                full_mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
                full_mask[int(y1):int(y2), int(x1):int(x2)] = mask
                series_masks.append({"mask": full_mask, "hue": hue_peak})
                
        return series_masks

    def _find_peaks(self, hist):
        peaks = []
        for i in range(1, 179):
            if hist[i] > 100 and hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peaks.append(i)
        return peaks
