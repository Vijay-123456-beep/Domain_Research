import cv2
import numpy as np
from skimage.morphology import skeletonize

class CurveTracer:
    """
    Traces continuous curves from binary masks using skeletonization.
    """
    def trace(self, mask):
        """
        Skeletonizes the mask and extracts ordered (x,y) coordinates.
        """
        # Convert to 0-1 for skimage
        skeleton = skeletonize(mask > 0).astype(np.uint8)
        
        # Find coordinates of the skeleton points
        points = np.column_stack(np.where(skeleton > 0))
        
        if len(points) == 0:
            return []
            
        # Order points by X-coordinate (assuming single-valued function y=f(x))
        # For more complex curves, a path-finding approach would be needed.
        ordered_points = sorted(points, key=lambda p: p[1]) # p[1] is x-pixel
        
        return [{"x": int(p[1]), "y": int(p[0])} for p in ordered_points]
