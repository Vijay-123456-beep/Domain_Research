import numpy as np
from scipy.interpolate import UnivariateSpline

class CurveSmoother:
    """
    Cleans, sorts, deduplicates, and smoothes raw (x,y) pixel-extracted scientific data points.
    Prevents noisy spikes from OCR or overlapping curve segmentations.
    """
    def __init__(self, smoothing_factor=0.5):
        self.smoothing_factor = smoothing_factor

    def clean_and_smooth(self, raw_points):
        """
        Takes raw dictionaries [{"x": float, "y": float}, ...]
        Returns a smoothed, monotonic dataset preserving the physical shape of the curve.
        """
        if not raw_points or len(raw_points) < 5:
            return raw_points
            
        # 1. Deduplication and conversion to numpy arrays
        # Use a dictionary to keep the highest Y value for any duplicate X (conservative for capacitance)
        unique_points = {}
        for p in raw_points:
            x_rnd = round(p["x"], 3)
            # If multiple markers hit same X, log the largest Y
            if x_rnd not in unique_points or p["y"] > unique_points[x_rnd]:
                unique_points[x_rnd] = p["y"]
                
        # 2. Sort by X (strict monotonic requirement for continuous plots)
        sorted_pairs = sorted(unique_points.items())
        x_arr = np.array([p[0] for p in sorted_pairs])
        y_arr = np.array([p[1] for p in sorted_pairs])
        
        if len(x_arr) < 5:
            return [{"x": float(x), "y": float(y)} for x, y in zip(x_arr, y_arr)]
            
        # 3. Outlier Removal (Local Median Filter)
        window_size = 5
        clean_mask = np.ones(len(y_arr), dtype=bool)
        for i in range(len(y_arr)):
            # Get local neighborhood
            start = max(0, i - window_size//2)
            end = min(len(y_arr), i + window_size//2 + 1)
            neighborhood = np.delete(y_arr[start:end], i - start)
            
            if len(neighborhood) == 0: continue
                
            local_median = np.median(neighborhood)
            local_std = np.std(neighborhood) if np.std(neighborhood) > 0 else 1e-5
            
            # If point is wildly far from its neighbors, flag as outlier spike
            if abs(y_arr[i] - local_median) > 3 * local_std and abs(y_arr[i] - local_median) > abs(local_median * 0.2):
                clean_mask[i] = False
                
        x_clean = x_arr[clean_mask]
        y_clean = y_arr[clean_mask]
        
        if len(x_clean) < 5:
            return [{"x": float(x), "y": float(y)} for x, y in zip(x_arr, y_arr)]
            
        # 4. Spline Interpolation (Smoothing)
        try:
            # s is the smoothing factor. We scale it by the number of points and variance
            variance = np.var(y_clean)
            s_val = len(x_clean) * variance * self.smoothing_factor
            
            spline = UnivariateSpline(x_clean, y_clean, s=s_val)
            y_smooth = spline(x_clean)
            
            # Re-format back to list of dicts
            result = [{"x": float(x), "y": float(y)} for x, y in zip(x_clean, y_smooth)]
            return result
            
        except Exception as e:
            print(f"  [Smoother] Spline fitting failed ({e}), returning raw sorted data.")
            return [{"x": float(x), "y": float(y)} for x, y in zip(x_clean, y_clean)]
