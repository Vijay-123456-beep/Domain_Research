import cv2
import numpy as np

class SubplotSplitter:
    """
    Detects and splits multi-panel figures (a, b, c, d) using whitespace segmentation.
    """
    def __init__(self, reader=None):
        self.reader = reader

    def split(self, img_np):
        """
        Detects subplots and returns a list of cropped images and their panel labels.
        Returns: [{"image": img, "label": "a", "bbox": [x,y,w,h]}, ...]
        """
        if img_np is None or img_np.size == 0:
            return []
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # 1. Whitespace Segmentation
        # Threshold to find white gaps
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        
        # Horizontal and Vertical Projections
        h_proj = np.sum(binary, axis=1)
        v_proj = np.sum(binary, axis=0)
        
        # Find continuous non-zero regions (potential subplots)
        h_regions = self._get_continuous_regions(h_proj, gap_threshold=10, noise_thresh=100*255)
        v_regions = self._get_continuous_regions(v_proj, gap_threshold=10, noise_thresh=30*255)
        
        subplots = []
        if len(h_regions) > 1 or len(v_regions) > 1:
            # Multi-panel detected
            for hr in h_regions:
                for vr in v_regions:
                    crop = img_np[hr[0]:hr[1], vr[0]:vr[1]]
                    if crop.size > 0:
                        label = self._detect_panel_label(crop)
                        subplots.append({
                            "image": crop,
                            "label": label,
                            "bbox": [vr[0], hr[0], vr[1]-vr[0], hr[1]-hr[0]]
                        })
        else:
            # Single panel
            subplots.append({
                "image": img_np,
                "label": "a",
                "bbox": [0, 0, img_np.shape[1], img_np.shape[0]]
            })
            
        return subplots

    def _get_continuous_regions(self, proj, gap_threshold=10, noise_thresh=0):
        regions = []
        start = None
        gap_count = 0
        
        for i, val in enumerate(proj):
            if val > noise_thresh:
                if start is None:
                    start = i
                gap_count = 0
            else:
                if start is not None:
                    gap_count += 1
                    if gap_count > gap_threshold:
                        regions.append((start, i - gap_count))
                        start = None
                        gap_count = 0
        if start is not None:
            regions.append((start, len(proj)))
        return regions

    def _detect_panel_label(self, crop):
        # Look for (a), (b), a), b) in the top-left or top-right corners
        if self.reader:
            # OCR the top-left corner primarily
            h, w = crop.shape[:2]
            corner = crop[0:int(h*0.2), 0:int(w*0.2)]
            # Guard: skip OCR if corner is empty or too small for reliable detection
            if corner is None or corner.size == 0 or corner.shape[0] < 5 or corner.shape[1] < 5:
                return "unknown"
            results = self.reader.readtext(corner)
            for res in results:
                text = res[1].lower().strip("(). ")
                if len(text) == 1 and text.isalpha():
                    return text
        return "unknown"
