import cv2
import numpy as np

class LegendMatcher:
    """
    Detects legend entries and matches text labels with curve colors.
    """
    def __init__(self, reader=None):
        self.reader = reader

    def match(self, img_np, series_colors):
        """
        Detects legend and returns {color_hue: label}.
        """
        if not self.reader: return {}
        
        # 1. OCR all text to find potential labels
        results = self.reader.readtext(img_np)
        
        mapping = {}
        for (bbox, text, prob) in results:
            # Check if this text box is near a color swatch
            # A color swatch is a small colored area
            x_min, y_min = np.min(bbox, axis=0)
            x_max, y_max = np.max(bbox, axis=0)
            
            # ROI to the left of the text for the swatch
            swatch_roi = img_np[int(y_min):int(y_max), max(0, int(x_min-20)):int(x_min)]
            if swatch_roi.size > 0:
                hsv_swatch = cv2.cvtColor(swatch_roi, cv2.COLOR_RGB2HSV)
                avg_hue = np.mean(hsv_swatch[:,:,0])
                
                # Match avg_hue to the closest hue in series_colors
                closest_hue = self._find_closest_hue(avg_hue, series_colors)
                if closest_hue is not None:
                    mapping[closest_hue] = text
                    
        return mapping

    def _find_closest_hue(self, hue, target_hues):
        if not target_hues: return None
        # Circular distance in HSV Hue (0-179)
        distances = [min(abs(hue - th), 180 - abs(hue - th)) for th in target_hues]
        if min(distances) < 15: # Threshold for matching
            return target_hues[np.argmin(distances)]
        return None
