import numpy as np

class ScaleCalibrator:
    """
    V2 Calibrator: Maps pixel coordinates to scientific values using robust linear/logarithmic regression.
    """
    def calibrate(self, axes_info, ticks, tick_values):
        """
        Calculates transformation functions (px -> value) for X and Y axes.
        Handles missing OCR values and inherently rejects outliers via RANSAC/Median scaling.
        """
        calib_info = {"x_func": None, "y_func": None, "x_scale_type": "linear", "y_scale_type": "linear", "valid": False}
        
        if not axes_info or not tick_values:
            return calib_info
            
        x_mapped = tick_values.get("x", {})
        y_mapped = tick_values.get("y", {})
        
        # We need at least 2 points to build a reliable scale function
        if len(x_mapped) >= 2:
            x_px, x_vals = self._filter_outliers(x_mapped)
            if len(x_px) >= 2:
                # Check if scale is logarithmic (values grow exponentially)
                if self._is_log_scale(x_px, x_vals):
                    calib_info["x_scale_type"] = "log"
                    # Fit log10(val) = m*px + b  ==> val = 10^(m*px + b)
                    log_vals = np.log10(np.abs(x_vals) + 1e-9) # Avoid log(0)
                    m, b = np.polyfit(x_px, log_vals, 1)
                    calib_info["x_func"] = lambda px, slope=m, intercept=b: 10**(slope * px + intercept)
                else:
                    m, b = np.polyfit(x_px, x_vals, 1)
                    calib_info["x_func"] = lambda px, slope=m, intercept=b: slope * px + intercept
                
        if len(y_mapped) >= 2:
            y_px, y_vals = self._filter_outliers(y_mapped)
            if len(y_px) >= 2:
                if self._is_log_scale(y_px, y_vals):
                    calib_info["y_scale_type"] = "log"
                    log_vals = np.log10(np.abs(y_vals) + 1e-9)
                    m, b = np.polyfit(y_px, log_vals, 1)
                    calib_info["y_func"] = lambda px, slope=m, intercept=b: 10**(slope * px + intercept)
                else:
                    m, b = np.polyfit(y_px, y_vals, 1)
                    # Note: Y pixel values go DOWN as scientific values go UP in images
                    calib_info["y_func"] = lambda px, slope=m, intercept=b: slope * px + intercept

        if calib_info["x_func"] and calib_info["y_func"]:
            calib_info["valid"] = True
            print(f"  [Calibration] Success (X:{calib_info['x_scale_type']}, Y:{calib_info['y_scale_type']}) based on {len(x_mapped)} X-ticks, {len(y_mapped)} Y-ticks")
        else:
            print("  [Calibration] Failed to establish reliable axes transforms.")
            
        # Store bounds for clamping later
        w, h = axes_info.get("x_axis_end", (1000, 0))[0], axes_info.get("origin", (0, 1000))[1]
        calib_info["x_bounds"] = (axes_info.get("origin", (0, 0))[0], w)
        calib_info["y_bounds"] = (axes_info.get("y_axis_end", (0, 0))[1], h)
        return calib_info

    def apply_calibration(self, points_px, calib_info):
        """Converts pixel points to scaled scientific values."""
        if not calib_info.get("valid"):
            # If calibration fails, return the raw pixels so the pipeline doesn't crash
            if points_px and isinstance(points_px[0], (list, tuple)):
                return [{"x": float(p[0]), "y": float(p[1])} for p in points_px]
            return points_px
            
        mapped = []
        x_func = calib_info["x_func"]
        y_func = calib_info["y_func"]
        min_x, max_x = calib_info["x_bounds"]
        min_y, max_y = calib_info["y_bounds"]
        
        for p in points_px:
            # Handle both list of tuples [(px,py)] and list of dicts [{"x":px, "y":py}]
            if isinstance(p, (tuple, list)):
                px, py = p[0], p[1]
            elif isinstance(p, dict):
                px, py = p["x"], p["y"]
            else:
                continue
                
            # Clamp coordinates to bounding box to prevent wild extrapolations
            c_px = max(min_x, min(px, max_x))
            c_py = max(min_y, min(py, max_y))
            
            x_val = x_func(c_px)
            y_val = y_func(c_py)
            mapped.append({"x": float(x_val), "y": float(y_val)})
            
        return mapped

    def _filter_outliers(self, mapped_dict):
        """Removes OCR outliers by checking distance uniformity."""
        pts = sorted(mapped_dict.items(), key=lambda i: i[0])
        px_arr = np.array([p[0] for p in pts])
        val_arr = np.array([p[1] for p in pts])
        
        if len(pts) < 3:
            return px_arr, val_arr
            
        # Compute local slopes (value delta / pixel delta)
        slopes = []
        for i in range(len(pts)-1):
            dp = px_arr[i+1] - px_arr[i]
            if dp == 0: continue
            slopes.append((val_arr[i+1] - val_arr[i]) / dp)
            
        if not slopes: return px_arr, val_arr
        
        med_slope = np.median(slopes)
        valid_idx = [0] # Always keep first point as anchor
        
        for i in range(1, len(pts)):
            dp = px_arr[i] - px_arr[valid_idx[-1]]
            expected_val = val_arr[valid_idx[-1]] + dp * med_slope
            actual_val = val_arr[i]
            
            # If actual value is within 20% of expected linear value, keep it
            if abs(actual_val - expected_val) < max(abs(expected_val * 0.2), 0.1):
                valid_idx.append(i)
                
        return px_arr[valid_idx], val_arr[valid_idx]

    def _is_log_scale(self, px_arr, val_arr):
        """Determines if a scale is logarithmic by checking if ratio of values is constant."""
        if len(val_arr) < 3: return False
        if any(v <= 0 for v in val_arr): return False # Logs can't be <= 0
        
        # Check if values grow by multiplier (e.g. 1, 10, 100) instead of addition
        ratios = [val_arr[i+1]/val_arr[i] for i in range(len(val_arr)-1)]
        std_dev = np.std(ratios)
        mean_ratio = np.mean(ratios)
        
        # If the ratio is very constant (low std dev) and > 1.5, it's likely logarithmic
        if mean_ratio > 1.5 and (std_dev / mean_ratio) < 0.2:
             return True
        return False
