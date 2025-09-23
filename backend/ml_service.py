import cv2
import numpy as np
from ultralytics import YOLO
import logging
from pathlib import Path
import torch
from typing import List, Tuple, Dict, Any
import time
import shutil
import os

logger = logging.getLogger(__name__)

class ModelManager:
    """Manage multiple YOLO models"""
    
    def __init__(self):
        self.models = {}
        self.active_model_name = "yolo11n"
        self.model_dir = Path("/tmp/custom_models")
        self.model_dir.mkdir(exist_ok=True)
    
    def load_model(self, model_path: str, model_name: str = None) -> str:
        """Load a YOLO model and return model name"""
        try:
            if model_name is None:
                model_name = Path(model_path).stem
            
            model = YOLO(model_path)
            model.to("cpu")  # Force CPU
            
            # Warm up the model
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = model.predict(dummy_image, device="cpu", verbose=False)
            
            self.models[model_name] = {
                "model": model,
                "path": model_path,
                "classes": list(model.names.values()),
                "loaded_at": time.time()
            }
            
            logger.info(f"Model {model_name} loaded successfully with {len(model.names)} classes")
            return model_name
            
        except Exception as e:
            logger.error(f"Error loading model {model_path}: {e}")
            raise e
    
    def switch_model(self, model_name: str):
        """Switch active model"""
        if model_name in self.models:
            self.active_model_name = model_name
            logger.info(f"Switched to model: {model_name}")
        else:
            raise ValueError(f"Model {model_name} not found")
    
    def get_active_model(self):
        """Get currently active model"""
        return self.models.get(self.active_model_name)
    
    def list_models(self) -> Dict[str, Any]:
        """List all loaded models"""
        return {name: {
            "classes": info["classes"],
            "path": info["path"],
            "loaded_at": info["loaded_at"],
            "is_active": name == self.active_model_name
        } for name, info in self.models.items()}

class LogoDetectionService:
    """
    CPU-optimized YOLO detection service with multi-model support
    """
    
    def __init__(self, model_path: str = "yolo11n.pt"):
        self.model_manager = ModelManager()
        self.device = "cpu"  # Force CPU for browser deployment
        self.confidence_threshold = 0.25
        self.iou_threshold = 0.45
        self.max_inference_size = 640  # Optimize for CPU performance
        self.bbox_thickness = 4  # Fixed bounding box thickness
        
        # Load default model
        self._load_default_model(model_path)
        
    def _load_default_model(self, model_path: str):
        """Load default YOLO model"""
        try:
            torch.set_num_threads(4)  # Optimize CPU threads
            logger.info(f"Loading default YOLO model: {model_path}")
            
            self.model_manager.load_model(model_path, "yolo11n")
            logger.info("Default model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading default model: {e}")
            raise e
    
    def upload_custom_model(self, model_file_path: str, model_name: str) -> Dict[str, Any]:
        """Upload and load a custom YOLO model"""
        try:
            # Validate model file
            if not Path(model_file_path).exists():
                raise FileNotFoundError("Model file not found")
            
            # Copy to custom models directory
            custom_model_path = self.model_manager.model_dir / f"{model_name}.pt"
            shutil.copy2(model_file_path, custom_model_path)
            
            # Load the model
            model_name = self.model_manager.load_model(str(custom_model_path), model_name)
            
            return {
                "success": True,
                "model_name": model_name,
                "classes": self.model_manager.models[model_name]["classes"],
                "message": f"Custom model {model_name} loaded successfully"
            }
            
        except Exception as e:
            logger.error(f"Error uploading custom model: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def switch_model(self, model_name: str) -> Dict[str, Any]:
        """Switch to a different model"""
        try:
            self.model_manager.switch_model(model_name)
            return {
                "success": True,
                "active_model": model_name,
                "message": f"Switched to model: {model_name}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_current_model(self):
        """Get current active model"""
        return self.model_manager.get_active_model()
    
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
    
    def process_video_file(self, video_path: str, output_path: str) -> Dict[str, Any]:
        """
        Process entire video file and return annotated video
        
        Args:
            video_path: Path to input video file
            output_path: Path for output annotated video
            
        Returns:
            Processing results with statistics
        """
        try:
            import cv2
            from pathlib import Path
            
            start_time = time.time()
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Could not open video file: {video_path}")
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            processed_frames = 0
            total_detections = 0
            frame_detections = []
            
            logger.info(f"Processing video: {total_frames} frames at {fps} FPS")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run detection on frame
                result = self.detect_logos_in_frame(frame)
                detections = result["detections"]
                annotated_frame = result["annotated_image"]
                
                # Write annotated frame
                out.write(annotated_frame)
                
                # Collect statistics
                processed_frames += 1
                total_detections += len(detections)
                frame_detections.append({
                    "frame": processed_frames,
                    "detections": len(detections),
                    "objects": [det["class_name"] for det in detections]
                })
                
                # Log progress every 30 frames
                if processed_frames % 30 == 0:
                    progress = (processed_frames / total_frames) * 100
                    logger.info(f"Processing progress: {progress:.1f}% ({processed_frames}/{total_frames})")
            
            # Clean up
            cap.release()
            out.release()
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "output_path": output_path,
                "processed_frames": processed_frames,
                "total_detections": total_detections,
                "processing_time": processing_time,
                "fps": fps,
                "resolution": f"{width}x{height}",
                "frame_detections": frame_detections[:10],  # First 10 frames for summary
                "avg_detections_per_frame": total_detections / processed_frames if processed_frames > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return {
                "success": False,
                "error": str(e),
                "processed_frames": processed_frames,
                "processing_time": time.time() - start_time if 'start_time' in locals() else 0
            }

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
            "bbox_thickness": self.bbox_thickness,
            "available_classes": list(self.model.names.values()) if self.model else []
        }