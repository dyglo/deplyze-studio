import cv2
import numpy as np
from ultralytics import YOLO
import logging
from pathlib import Path
import torch
from typing import List, Tuple, Dict, Any
import time

logger = logging.getLogger(__name__)

class LogoDetectionService:
    """
    CPU-optimized YOLO11n logo detection service
    """
    
    def __init__(self, model_path: str = "yolo11n.pt"):
        self.model_path = model_path
        self.model = None
        self.device = "cpu"  # Force CPU for browser deployment
        self.confidence_threshold = 0.25
        self.iou_threshold = 0.45
        self.max_inference_size = 640  # Optimize for CPU performance
        self.bbox_thickness = 2  # Default bounding box thickness
        
        # Load model at initialization
        self._load_model()
        
    def _load_model(self):
        """Load YOLO model with CPU optimization"""
        try:
            # Force CPU usage for browser deployment
            torch.set_num_threads(4)  # Optimize CPU threads
            
            logger.info(f"Loading YOLO model: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # Force model to CPU
            self.model.to(self.device)
            
            # Warm up the model with a dummy prediction
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model.predict(dummy_image, device=self.device, verbose=False)
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e
    
    def detect_logos_in_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect logos in a single image
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            Detection results with bounding boxes and labels
        """
        try:
            start_time = time.time()
            
            # Run inference
            results = self.model.predict(
                image,
                device=self.device,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.max_inference_size,
                verbose=False
            )
            
            # Process results
            detections = []
            annotated_image = image.copy()
            
            if results and len(results) > 0:
                result = results[0]  # First result
                
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
                    confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
                    class_ids = result.boxes.cls.cpu().numpy()  # Class IDs
                    
                    # Draw bounding boxes and collect detections
                    for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Get class name (for now using class_id, later we'll map to logo brands)
                        class_name = self.model.names.get(int(class_id), f"Logo_{int(class_id)}")
                        
                        # Add detection
                        detections.append({
                            "id": i,
                            "class_name": class_name,
                            "confidence": float(conf),
                            "bbox": {
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2,
                                "width": x2 - x1,
                                "height": y2 - y1
                            }
                        })
                        
                        # Draw bounding box
                        color = self._get_color_for_class(class_id)
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, self.bbox_thickness)
                        
                        # Draw label
                        label = f"{class_name}: {conf:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), color, -1)
                        cv2.putText(annotated_image, label, (x1, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            inference_time = time.time() - start_time
            
            return {
                "detections": detections,
                "inference_time": inference_time,
                "annotated_image": annotated_image,
                "original_shape": image.shape[:2],  # (height, width)
                "model_input_size": self.max_inference_size
            }
            
        except Exception as e:
            logger.error(f"Error in logo detection: {e}")
            return {
                "detections": [],
                "inference_time": 0.0,
                "annotated_image": image,
                "error": str(e)
            }
    
    def detect_logos_in_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Optimized detection for video frames (lighter processing)
        """
        # For video, we can use smaller input size for better FPS
        original_size = self.max_inference_size
        self.max_inference_size = 416  # Smaller for video processing
        
        result = self.detect_logos_in_image(frame)
        
        # Restore original size
        self.max_inference_size = original_size
        
        return result
    
    def _get_color_for_class(self, class_id: int) -> Tuple[int, int, int]:
        """Generate consistent color for each class"""
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green  
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
        ]
        return colors[int(class_id) % len(colors)]
    
    def update_confidence_threshold(self, threshold: float):
        """Update confidence threshold for detections"""
        self.confidence_threshold = max(0.1, min(0.9, threshold))
        logger.info(f"Updated confidence threshold to {self.confidence_threshold}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "max_inference_size": self.max_inference_size,
            "available_classes": list(self.model.names.values()) if self.model else []
        }