import cv2
import numpy as np

class MarkerDetector:
    """
    Detects discrete markers (circles, squares, triangles) in the plot.
    """
    def detect(self, img_np, axes_info):
        """
        Returns markers grouped by shape and color.
        """
        if not axes_info: return []
        if img_np is None or img_np.size == 0: return []
        
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Crop to axes
        x1, y1, x2, y2 = axes_info["bbox"]
        roi = binary[int(y1):int(y2), int(x1):int(x2)]
        
        # Find connected components (potential markers)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi)
        
        markers = []
        for i in range(1, num_labels):
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Filter by size typical for markers
            if 4 < w < 20 and 4 < h < 20:
                cx, cy = centroids[i]
                
                # Basic shape identification
                circularity = 4 * np.pi * area / (2 * (w + h))**2
                shape = "circle" if circularity > 0.7 else "square/triangle"
                
                # Get color from original image
                color = img_np[int(y1 + cy), int(x1 + cx)].tolist()
                
                markers.append({
                    "pos": (int(x1 + cx), int(y1 + cy)),
                    "shape": shape,
                    "color": color
                })
                
        return markers
