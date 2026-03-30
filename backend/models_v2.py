import os
import torch
import numpy as np
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
    from torchvision.transforms import functional as F
except ImportError:
    maskrcnn_resnet50_fpn_v2 = None

class ModelManagerV2:
    """
    Unified manager for lazy-loading heavy Deep Learning models for Plot Extraction V2.
    Ensures models are only loaded into VRAM when actually needed.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManagerV2, cls).__new__(cls)
            cls._instance.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            cls._instance.yolo_model = None
            cls._instance.mask_rcnn_model = None
        return cls._instance

    def get_yolo(self):
        """Lazy load YOLOv8n object detection model."""
        if self.yolo_model is None:
            if YOLO is None:
                raise ImportError("ultralytics package is required for YOLOv8. Run: pip install ultralytics")
                
            yolo_path = os.path.join(os.path.dirname(__file__), 'yolov8n.pt')
            if not os.path.exists(yolo_path) and not os.path.exists('yolov8n.pt'):
                raise RuntimeError(f"Offline Mode: YOLO weights not found at {yolo_path}. Skipping AI Edge Detection.")
                
            print(f"[MODELS_V2] Loading YOLOv8n onto {self.device}...")
            self.yolo_model = YOLO(yolo_path if os.path.exists(yolo_path) else 'yolov8n.pt')
            self.yolo_model.to(self.device)
        return self.yolo_model

    def get_mask_rcnn(self):
        """Lazy load Mask R-CNN instance segmentation model."""
        if self.mask_rcnn_model is None:
            if maskrcnn_resnet50_fpn_v2 is None:
                raise ImportError("torchvision package is required for Mask R-CNN. Run: pip install torchvision")
                
            # Check for local Torch cache (prevents huggingface/torchvision hang offline)
            cache_dir = os.path.expanduser('~/.cache/torch/hub/checkpoints')
            weights_name = "maskrcnn_resnet50_fpn_v2"
            has_cache = os.path.exists(cache_dir) and any(weights_name in f for f in os.listdir(cache_dir))
            
            local_weights = os.path.join(os.path.dirname(__file__), f'{weights_name}.pth')
            if not has_cache and not os.path.exists(local_weights):
                raise RuntimeError("Offline Mode: Mask R-CNN weights not cached locally. Skipping AI Curve Segmentation.")
                
            print(f"[MODELS_V2] Loading Mask R-CNN onto {self.device}...")
            weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.mask_rcnn_model = maskrcnn_resnet50_fpn_v2(weights=weights)
            self.mask_rcnn_model.eval() # Set to inference mode
            self.mask_rcnn_model.to(self.device)
            self.mask_rcnn_transform = weights.transforms()
        return self.mask_rcnn_model, self.mask_rcnn_transform


class YOLOEdgeDetector:
    """
    Replaces AxisDetector and SubplotSplitter using YOLOv8 bounding boxes.
    """
    def __init__(self):
        self.manager = ModelManagerV2()
        
    def detect_plot_area(self, img_np):
        """Uses YOLO to find the primary plot area (axes bounding box)."""
        model = self.manager.get_yolo()
        
        # YOLO inference
        results = model(img_np, verbose=False)[0]
        
        best_box = None
        max_area = 0
        
        # Find the largest bounding box (likely the main plot axes)
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                best_box = [x1, y1, x2, y2]
                
        if best_box is None:
            # Fallback to whole image margin if YOLO fails
            h, w = img_np.shape[:2]
            return {"bbox": [w*0.1, h*0.1, w*0.9, h*0.9], "origin": (w*0.1, h*0.9), "x_axis_end": (w*0.9, h*0.9), "y_axis_end": (w*0.1, h*0.1)}
            
        x1, y1, x2, y2 = best_box
        return {
            "bbox": [x1, y1, x2, y2],
            "origin": (x1, y2),
            "x_axis_end": (x2, y2),
            "y_axis_end": (x1, y1)
        }


class CurveMaskRCNN:
    """
    Replaces CurveSegmenter's HSV heuristics with Mask R-CNN Instance Segmentation.
    Handles overlapping curves, dashed lines, and grayscale natively.
    """
    def __init__(self):
        self.manager = ModelManagerV2()
        
    def segment(self, img_np, axes_info):
        """Returns a list of binary masks for each detected curve."""
        if img_np is None or img_np.size == 0 or not axes_info:
            return []
            
        model, transform = self.manager.get_mask_rcnn()
        device = self.manager.device
        
        x1, y1, x2, y2 = axes_info["bbox"]
        roi = img_np[int(y1):int(y2), int(x1):int(x2)]
        if roi.size == 0:
            return []
            
        # Convert RGB numpy to PyTorch tensor
        img_tensor = torch.from_numpy(roi).permute(2, 0, 1).float() / 255.0
        
        with torch.no_grad():
            prediction = model([img_tensor.to(device)])[0]
            
        masks = prediction['masks']
        scores = prediction['scores']
        
        series_masks = []
        # Filter masks by confidence threshold (> 0.5)
        for i in range(len(scores)):
            if scores[i] > 0.5:
                # Squeeze channel dim and convert to binary numpy mask
                mask_np = masks[i, 0].cpu().numpy()
                binary_mask = (mask_np > 0.5).astype(np.uint8) * 255
                
                if cv2.countNonZero(binary_mask) > 50:
                    # Map ROI mask back to full image coordinates
                    full_mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
                    full_mask[int(y1):int(y2), int(x1):int(x2)] = binary_mask
                    
                    # Extract average color from the masked region for legend matching
                    mean_color = cv2.mean(img_np, mask=full_mask)[:3]
                    
                    series_masks.append({
                        "mask": full_mask,
                        "color_signature": mean_color,
                        "confidence": float(scores[i])
                    })
                    
        return series_masks
