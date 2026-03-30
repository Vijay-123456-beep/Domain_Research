import cv2
import numpy as np

class AxisDetector:
    """
    Detects plot axes using Hough Line Transform and perpendicularity checks.
    """
    def detect(self, img_np):
        """
        Returns the bounding box of the axes [x_min, y_min, x_max, y_max].
        """
        if img_np is None or img_np.size == 0:
            return None
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Use probabilistic Hough transform for segments
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=100, maxLineGap=20)
        
        if lines is None:
            return None
            
        horizontals = []
        verticals = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            if abs(y1 - y2) < 5: # Horizontal
                horizontals.append(line[0])
            elif abs(x1 - x2) < 5: # Vertical
                verticals.append(line[0])
                
        if not horizontals or not verticals:
            return None
            
        # Strategy: X-axis is usually the bottom-most long horizontal. 
        # Y-axis is usually the left-most long vertical.
        x_axis = sorted(horizontals, key=lambda l: max(l[1], l[3]), reverse=True)[0]
        y_axis = sorted(verticals, key=lambda l: min(l[0], l[2]))[0]
        
        # Intersection or near-intersection point
        origin_x = y_axis[0]
        origin_y = x_axis[1]
        
        x_max = max(x_axis[0], x_axis[2])
        y_max = min(y_axis[1], y_axis[3]) # Y decreases going up in images
        
        return {
            "origin": (origin_x, origin_y),
            "x_axis_end": (x_max, origin_y),
            "y_axis_end": (origin_x, y_max),
            "bbox": [origin_x, y_max, x_max, origin_y] # [left, top, right, bottom]
        }
