import cv2
import numpy as np

class TickDetector:
    """
    Detects tick marks along the horizontal and vertical axes.
    """
    def detect(self, img_np, axes_info):
        """
        Returns clusters of tick positions for X and Y axes.
        """
        if not axes_info: return None
        if img_np is None or img_np.size == 0:
            return None
        
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        origin_x, origin_y = axes_info["origin"]
        x_end_x, _ = axes_info["x_axis_end"]
        _, y_end_y = axes_info["y_axis_end"]
        
        # 1. Look for vertical segments along the x-axis
        x_ticks = []
        x_axis_roi = edges[origin_y-10:origin_y+10, origin_x:x_end_x]
        # Use findContours to find small vertical segments
        contours, _ = cv2.findContours(x_axis_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > 3 and w < 5: # Small vertical segment
                x_ticks.append(origin_x + x + w//2)
                
        # 2. Look for horizontal segments along the y-axis
        y_ticks = []
        y_axis_roi = edges[y_end_y:origin_y, origin_x-10:origin_x+10]
        contours, _ = cv2.findContours(y_axis_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 3 and h < 5: # Small horizontal segment
                y_ticks.append(y_end_y + y + h//2)
                
        return {
            "x_ticks": sorted(list(set(x_ticks))),
            "y_ticks": sorted(list(set(y_ticks)))
        }
