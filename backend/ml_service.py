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
import base64

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

    def get_model(self, model_name: str = None):
        if model_name and model_name in self.models:
            return self.models[model_name]
        return self.get_active_model()
    
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
        self.bbox_thickness = 6  # Thicker bounding boxes for clarity
        
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

    def get_named_model(self, model_name: str = None):
        return self.model_manager.get_model(model_name)
    
    def detect_logos_in_image(self, image: np.ndarray, selected_classes: List[str] = None) -> Dict[str, Any]:
        """
        Detect objects in a single image with detailed classification and confidence
        
        Args:
            image: OpenCV image (BGR format)
            selected_classes: List of class names to filter detections (optional)
            
        Returns:
            Detection results with detailed classification info
        """
        try:
            start_time = time.time()
            
            # Get current model
            current_model = self.get_current_model()
            if not current_model:
                raise Exception("No model loaded")
            
            model = current_model["model"]
            
            # Run inference
            results = model.predict(
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
            total_confidence = 0
            
            if results and len(results) > 0:
                result = results[0]  # First result
                
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
                    confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
                    class_ids = result.boxes.cls.cpu().numpy()  # Class IDs
                    
                    # Draw bounding boxes and collect detections
                    for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Get class name
                        class_name = model.names.get(int(class_id), f"Object_{int(class_id)}")
                        
                        # Apply class filtering if specified
                        if selected_classes and class_name not in selected_classes:
                            continue  # Skip this detection if not in selected classes
                        
                        # Enhanced detection info
                        detection_info = {
                            "id": len(detections),  # Use filtered index
                            "class_name": class_name,
                            "class_id": int(class_id),
                            "confidence": float(conf),
                            "confidence_percentage": round(float(conf) * 100, 2),
                            "bbox": {
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2,
                                "width": x2 - x1,
                                "height": y2 - y1,
                                "center_x": (x1 + x2) // 2,
                                "center_y": (y1 + y2) // 2,
                                "area": (x2 - x1) * (y2 - y1)
                            },
                            "reliability": self._get_reliability_level(float(conf))
                        }
                        
                        detections.append(detection_info)
                        total_confidence += float(conf)
                        
                        # Draw bounding box with enhanced styling
                        color = self._get_color_for_class(class_id)
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, self.bbox_thickness)
                        
                        # Enhanced label with confidence
                        label = f"{class_name}: {conf:.2%}"
                        label_bg_color = color
                        
                        # Calculate label dimensions
                        font_scale = 0.6
                        font_thickness = 2
                        (label_width, label_height), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                        )
                        
                        # Draw label background
                        cv2.rectangle(annotated_image, 
                                    (x1, y1 - label_height - baseline - 10), 
                                    (x1 + label_width + 10, y1), 
                                    label_bg_color, -1)
                        
                        # Draw label text
                        cv2.putText(annotated_image, label, (x1 + 5, y1 - baseline - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
                        
                        # Add confidence indicator (small rectangle)
                        confidence_width = int((x2 - x1) * conf)
                        cv2.rectangle(annotated_image, (x1, y2 + 2), (x1 + confidence_width, y2 + 6), color, -1)
            
            inference_time = time.time() - start_time
            
            # Calculate statistics
            avg_confidence = total_confidence / len(detections) if detections else 0
            
            return {
                "detections": detections,
                "summary": {
                    "total_objects": len(detections),
                    "average_confidence": round(avg_confidence, 3),
                    "highest_confidence": max([d["confidence"] for d in detections]) if detections else 0,
                    "object_classes": list(set([d["class_name"] for d in detections])),
                    "inference_time": inference_time,
                    "model_used": self.model_manager.active_model_name,
                    "filtered_classes": selected_classes or "All classes",
                    "total_classes_available": len(current_model["classes"])
                },
                "inference_time": inference_time,
                "annotated_image": annotated_image,
                "original_shape": image.shape[:2],  # (height, width)
                "model_input_size": self.max_inference_size
            }
            
        except Exception as e:
            logger.error(f"Error in object detection: {e}")
            return {
                "detections": [],
                "summary": {"error": str(e)},
                "inference_time": 0.0,
                "annotated_image": image,
                "error": str(e)
            }
    
    def process_batch_images(self, image_paths: List[str], selected_classes: List[str] = None) -> Dict[str, Any]:
        """
        Process multiple images in batch
        
        Args:
            image_paths: List of image file paths
            selected_classes: List of class names to filter detections (optional)
            
        Returns:
            Batch processing results
        """
        try:
            start_time = time.time()
            batch_results = []
            total_detections = 0
            
            for i, image_path in enumerate(image_paths):
                try:
                    # Load image
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    
                    # Process image
                    result = self.detect_logos_in_image(image, selected_classes)
                    
                    # Save annotated image
                    output_path = f"/tmp/batch_results/annotated_{i}_{Path(image_path).name}"
                    Path(output_path).parent.mkdir(exist_ok=True)
                    cv2.imwrite(output_path, result["annotated_image"])
                    
                    batch_results.append({
                        "original_path": image_path,
                        "annotated_path": output_path,
                        "detections": result["detections"],
                        "summary": result.get("summary", {}),
                        "index": i
                    })
                    
                    total_detections += len(result["detections"])
                    
                except Exception as e:
                    logger.error(f"Error processing image {image_path}: {e}")
                    batch_results.append({
                        "original_path": image_path,
                        "error": str(e),
                        "index": i
                    })
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "batch_results": batch_results,
                "summary": {
                    "total_images": len(image_paths),
                    "processed_images": len([r for r in batch_results if "error" not in r]),
                    "failed_images": len([r for r in batch_results if "error" in r]),
                    "total_detections": total_detections,
                    "processing_time": processing_time,
                    "avg_time_per_image": processing_time / len(image_paths) if image_paths else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return {
                "success": False,
                "error": str(e),
                "batch_results": []
            }
    
    def _get_reliability_level(self, confidence: float) -> str:
        """Get reliability level based on confidence score"""
        if confidence >= 0.8:
            return "High"
        elif confidence >= 0.6:
            return "Medium"
        elif confidence >= 0.4:
            return "Low"
        else:
            return "Very Low"
    
    def detect_logos_in_frame(self, frame: np.ndarray, selected_classes: List[str] = None) -> Dict[str, Any]:
        """
        Optimized detection for video frames (lighter processing) with class filtering
        """
        # For video, we can use smaller input size for better FPS
        original_size = self.max_inference_size
        self.max_inference_size = 416  # Smaller for video processing
        
        result = self.detect_logos_in_image(frame, selected_classes)
        
        # Restore original size
        self.max_inference_size = original_size
        
        return result
    
    def _get_color_for_class(self, class_id: int) -> Tuple[int, int, int]:
        """Generate consistent color for each class based on brand palette"""
        # BGR Format: (B, G, R)
        colors = [
            (60, 93, 198),   # Crail (#C65D3C) - Brand Primary
            (158, 166, 175), # Cloudy (#AFA69E) - Brand Neutral
            (40, 40, 40),    # Dark Grey
            (60, 120, 198),  # Crail-ish lighter
            (93, 198, 60),   # Green-ish
            (198, 60, 93),   # Pink-ish
            (198, 160, 60),  # Yellow-ish
            (120, 60, 198),  # Purple-ish
        ]
        return colors[int(class_id) % len(colors)]

    def _get_color_for_track(self, track_id: int) -> Tuple[int, int, int]:
        colors = [
            (60, 93, 198),
            (158, 166, 175),
            (40, 40, 40),
            (60, 120, 198),
            (93, 198, 60),
            (198, 60, 93),
            (198, 160, 60),
            (120, 60, 198),
        ]
        return colors[int(track_id) % len(colors)]
    
    def process_video_file(self, video_path: str, output_path: str, selected_classes: List[str] = None) -> Dict[str, Any]:
        """
        Process entire video file and return annotated video with class filtering
        
        Args:
            video_path: Path to input video file
            output_path: Path for output annotated video
            selected_classes: List of class names to filter detections (optional)
            
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
            unique_detections = {} # class_name: {count, total_conf}
            frame_detections = []
            
            logger.info(f"Processing video: {total_frames} frames at {fps} FPS with class filter: {selected_classes or 'All classes'}")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run detection on frame with class filtering
                result = self.detect_logos_in_frame(frame, selected_classes)
                detections = result["detections"]
                annotated_frame = result["annotated_image"]
                
                # Write annotated frame
                out.write(annotated_frame)
                
                # Collect statistics
                processed_frames += 1
                total_detections += len(detections)
                
                # Accrue unique detections
                for det in detections:
                    cls_name = det["class_name"]
                    if cls_name not in unique_detections:
                        unique_detections[cls_name] = {"count": 0, "total_conf": 0}
                    unique_detections[cls_name]["count"] += 1
                    unique_detections[cls_name]["total_conf"] += det["confidence"]

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
                "detected_classes": [
                    {"class_name": cls, "count": data["count"], "avg_confidence": data["total_conf"] / data["count"]}
                    for cls, data in unique_detections.items()
                ],
                "frame_detections": frame_detections[:10],  # First 10 frames for summary
                "avg_detections_per_frame": total_detections / processed_frames if processed_frames > 0 else 0,
                "filtered_classes": selected_classes or "All classes"
            }

        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return {
                "success": False,
                "error": str(e),
                "processed_frames": processed_frames,
                "processing_time": time.time() - start_time if 'start_time' in locals() else 0
            }

    def track_video_file(self, video_path: str, output_path: str, model_name: str = None, tracker: str = "botsort") -> Dict[str, Any]:
        try:
            start_time = time.time()
            current_model = self.get_named_model(model_name)
            if not current_model:
                raise RuntimeError("No model loaded")

            model = current_model["model"]
            tracker_config = "botsort.yaml" if tracker == "botsort" else "bytetrack.yaml"

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video file: {video_path}")

            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 24
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            processed_frames = 0
            total_detections = 0
            frame_results = []
            unique_track_ids = set()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.track(
                    source=frame,
                    persist=True,
                    device=self.device,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    imgsz=416,
                    verbose=False,
                    tracker=tracker_config,
                )
                result = results[0] if results else None
                annotated_frame = frame.copy()
                detections = []

                if result is not None and result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy()
                    track_ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else [None] * len(boxes)

                    for box, conf, class_id, track_id in zip(boxes, confidences, class_ids, track_ids):
                        x1, y1, x2, y2 = map(int, box)
                        class_name = model.names.get(int(class_id), f"Object_{int(class_id)}")
                        color = self._get_color_for_track(track_id if track_id is not None else int(class_id))
                        label = f"#{track_id} {class_name} {conf:.2%}" if track_id is not None else f"{class_name} {conf:.2%}"

                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, self.bbox_thickness)
                        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                        cv2.rectangle(annotated_frame, (x1, max(0, y1 - label_height - baseline - 8)), (x1 + label_width + 8, y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                        detection = {
                            "class_name": class_name,
                            "class_id": int(class_id),
                            "confidence": float(conf),
                            "track_id": int(track_id) if track_id is not None else None,
                            "bbox": {
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2,
                                "width": x2 - x1,
                                "height": y2 - y1,
                            },
                        }
                        detections.append(detection)
                        if track_id is not None:
                            unique_track_ids.add(int(track_id))

                out.write(annotated_frame)
                processed_frames += 1
                total_detections += len(detections)
                frame_results.append({
                    "frame_index": processed_frames,
                    "detections": detections,
                })

                if processed_frames % 30 == 0:
                    progress = (processed_frames / max(total_frames, 1)) * 100
                    logger.info(f"Tracking progress: {progress:.1f}% ({processed_frames}/{total_frames})")

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
                "avg_detections_per_frame": total_detections / processed_frames if processed_frames else 0,
                "tracker": tracker,
                "model_used": model_name or self.model_manager.active_model_name,
                "total_tracks": len(unique_track_ids),
                "frame_results": frame_results,
            }
        except Exception as e:
            logger.error(f"Error tracking video: {e}")
            return {
                "success": False,
                "error": str(e),
                "processed_frames": 0,
                "processing_time": time.time() - start_time if 'start_time' in locals() else 0,
            }

    def convert_pt_to_onnx(self, model_file_path: str, task: str = "detect") -> Dict[str, Any]:
        try:
            if not Path(model_file_path).exists():
                raise FileNotFoundError("Model file not found")

            export_root = Path("/tmp/converted_models")
            export_root.mkdir(exist_ok=True, parents=True)
            model = YOLO(model_file_path)
            exported_path = model.export(format="onnx", imgsz=640 if task != "classify" else 224, device="cpu", simplify=False)
            exported_path = Path(exported_path)
            target_path = export_root / exported_path.name
            if exported_path.resolve() != target_path.resolve():
              shutil.copy2(exported_path, target_path)
            else:
              target_path = exported_path

            encoded = base64.b64encode(target_path.read_bytes()).decode("utf-8")
            return {
                "success": True,
                "onnx_model": encoded,
                "filename": target_path.name,
            }
        except Exception as e:
            logger.error(f"Error converting model to ONNX: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def update_confidence_threshold(self, threshold: float):
        """Update confidence threshold for detections"""
        self.confidence_threshold = max(0.1, min(0.9, threshold))
        logger.info(f"Updated confidence threshold to {self.confidence_threshold}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        current_model = self.get_current_model()
        return {
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "max_inference_size": self.max_inference_size,
            "bbox_thickness": self.bbox_thickness,
            "active_model": self.model_manager.active_model_name,
            "available_models": self.model_manager.list_models(),
            "available_classes": current_model["classes"] if current_model else []
        }
