from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response, FileResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime, timezone
import cv2
import numpy as np
import base64
import json
import asyncio
from io import BytesIO
from PIL import Image
import tempfile
import shutil

# Import our ML service
from ml_service import LogoDetectionService
from roboflow_service import RoboflowDatasetService

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Initialize ML service at startup
ml_service = LogoDetectionService()
roboflow_service = RoboflowDatasetService(os.environ.get("ROBOFLOW_API_KEY"))

# Create the main app without a prefix
app = FastAPI(title="Deplyze Studio API", version="1.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Pydantic Models
class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    width: int
    height: int

class Detection(BaseModel):
    id: int
    class_name: str
    confidence: float
    bbox: BoundingBox

class DetectionResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    detections: List[Detection]
    inference_time: float
    original_shape: List[int]  # [height, width]
    model_input_size: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = None

class DetectionHistory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    detection_result: DetectionResult
    image_name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str


def get_allowed_origins() -> List[str]:
    """Build a local-development-friendly CORS allowlist."""
    configured_origins = [
        origin.strip()
        for origin in os.environ.get("CORS_ORIGINS", "").split(",")
        if origin.strip()
    ]
    default_local_origins = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ]

    if "*" in configured_origins:
        return default_local_origins

    return list(dict.fromkeys(default_local_origins + configured_origins))

class ModelInfo(BaseModel):
    device: str
    confidence_threshold: float
    iou_threshold: float
    max_inference_size: int
    bbox_thickness: int
    active_model: str
    available_models: Dict[str, Any]
    available_classes: List[str]

class BatchProcessingResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    success: bool
    total_images: int
    processed_images: int
    failed_images: int
    total_detections: int
    processing_time: float
    batch_results: List[Dict[str, Any]] = []
    results_archive: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = None

class VideoProcessingResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    success: bool
    processed_frames: int
    total_detections: int
    processing_time: float
    fps: int
    resolution: str
    avg_detections_per_frame: float
    output_filename: str
    detected_classes: List[Dict[str, Any]] = []
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = None

class RoboflowInferRequest(BaseModel):
    dataset_id: str
    workspace: str
    project: str
    version: Optional[int] = None
    task: str
    model: str
    api_key: Optional[str] = None

class TrackRequest(BaseModel):
    video: str
    model: Optional[str] = None
    tracker: str = "botsort"

class TrackingProcessingResult(BaseModel):
    success: bool
    processed_frames: int
    total_detections: int
    processing_time: float
    fps: int
    resolution: str
    avg_detections_per_frame: float
    output_filename: str
    tracker: str
    model_used: str
    total_tracks: int
    frame_results: List[Dict[str, Any]] = []
    error: Optional[str] = None

# Helper functions
def numpy_to_base64(image_array: np.ndarray) -> str:
    """Convert numpy array to base64 string"""
    _, buffer = cv2.imencode('.jpg', image_array)
    return base64.b64encode(buffer).decode('utf-8')

def base64_to_numpy(base64_string: str) -> np.ndarray:
    """Convert base64 string to numpy array"""
    image_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def pil_to_opencv(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL image to OpenCV format"""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Deplyze Studio API is running!", "version": "1.0.0"}

@api_router.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get current model information"""
    model_info = ml_service.get_model_info()
    return ModelInfo(**model_info)

@api_router.get("/roboflow/search")
async def search_roboflow_datasets(q: str, api_key: Optional[str] = None):
    try:
        resolved_key = roboflow_service.resolve_api_key(api_key)
        return {"results": roboflow_service.search_public_datasets(q, resolved_key)}
    except Exception as e:
        logging.error(f"Error searching Roboflow datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/roboflow/dataset/{workspace}/{project}")
async def get_roboflow_dataset(workspace: str, project: str, api_key: Optional[str] = None):
    try:
        resolved_key = roboflow_service.resolve_api_key(api_key)
        return roboflow_service.dataset_detail(workspace, project, resolved_key)
    except Exception as e:
        logging.error(f"Error getting Roboflow dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/roboflow/infer")
async def infer_roboflow_dataset(payload: RoboflowInferRequest):
    try:
        resolved_key = roboflow_service.resolve_api_key(payload.api_key)
        samples = roboflow_service.inferable_samples(payload.workspace, payload.project, payload.version, resolved_key)
        if payload.task != "detect":
            return {
                "task": payload.task,
                "model": payload.model,
                "sample_images": samples,
            }

        results = []
        for sample in samples:
          cv_image = base64_to_numpy(sample["image_base64"])
          detect_result = ml_service.detect_logos_in_image(cv_image)
          results.append({
              "sample_id": sample["sample_id"],
              "file_name": sample["file_name"],
              "image_url": sample["image_url"],
              "inference_time": detect_result["inference_time"],
              "detections": detect_result["detections"],
          })

        return {
            "task": payload.task,
            "model": payload.model,
            "sample_images": samples,
            "results": results,
        }
    except Exception as e:
        logging.error(f"Error running Roboflow inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/model/confidence")
async def update_confidence_threshold(threshold: float):
    """Update model confidence threshold"""
    if not 0.1 <= threshold <= 0.9:
        raise HTTPException(status_code=400, detail="Threshold must be between 0.1 and 0.9")
    
    ml_service.update_confidence_threshold(threshold)
    return {"message": f"Confidence threshold updated to {threshold}"}

@api_router.get("/download/image/{image_id}")
async def download_annotated_image(image_id: str):
    """
    Download annotated image by ID
    """
    try:
        # Get detection result from database
        detection = await db.detection_history.find_one({"id": image_id})
        if not detection:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Check if we have cached annotated image
        image_path = Path(f"/tmp/annotated_images/{image_id}.jpg")
        
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Annotated image file not found")
        
        return FileResponse(
            path=str(image_path),
            filename=f"deplyze_studio_annotated_{image_id}.jpg",
            media_type="image/jpeg"
        )
        
    except Exception as e:
        logging.error(f"Error downloading annotated image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/model/upload")
async def upload_custom_model(background_tasks: BackgroundTasks, file: UploadFile = File(...), model_name: str = "custom_model"):
    """
    Upload a custom YOLO model
    """
    try:
        # Validate file type
        if not file.filename.endswith(('.pt', '.onnx')):
            raise HTTPException(status_code=400, detail="Model file must be .pt or .onnx format")
        
        # Create temporary file
        temp_path = f"/tmp/{uuid.uuid4().hex}_{file.filename}"
        
        with open(temp_path, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)
        
        # Upload model
        result = ml_service.upload_custom_model(temp_path, model_name)
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_file, temp_path)
        
        if result["success"]:
            return {
                "success": True,
                "model_name": result["model_name"],
                "classes": result["classes"],
                "message": result["message"]
            }
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except Exception as e:
        logging.error(f"Error uploading custom model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/model/convert")
async def convert_custom_model(background_tasks: BackgroundTasks, file: UploadFile = File(...), task: str = "detect"):
    try:
        if not file.filename.endswith(".pt"):
            raise HTTPException(status_code=400, detail="Only .pt files can be converted")

        temp_path = f"/tmp/{uuid.uuid4().hex}_{file.filename}"
        with open(temp_path, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)

        result = ml_service.convert_pt_to_onnx(temp_path, task=task)
        background_tasks.add_task(cleanup_file, temp_path)

        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])

        return {"onnx_model": result["onnx_model"], "filename": result["filename"]}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error converting custom model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/model/switch")
async def switch_model(model_name: str):
    """
    Switch to a different model
    """
    try:
        result = ml_service.switch_model(model_name)
        
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except Exception as e:
        logging.error(f"Error switching model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/detect/batch/images", response_model=BatchProcessingResult)
async def process_batch_images(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """
    Process multiple images in batch
    """
    try:
        if len(files) > 50:  # Limit batch size
            raise HTTPException(status_code=400, detail="Maximum 50 images per batch")
        
        # Create temp directory for batch
        batch_dir = Path(f"/tmp/batch_{uuid.uuid4().hex}")
        batch_dir.mkdir(exist_ok=True)
        
        # Save uploaded files
        image_paths = []
        for i, file in enumerate(files):
            if not file.content_type.startswith('image/'):
                continue
            
            file_path = batch_dir / f"image_{i}_{file.filename}"
            with open(file_path, "wb") as buffer:
                contents = await file.read()
                buffer.write(contents)
            
            image_paths.append(str(file_path))
        
        # Process batch
        result = ml_service.process_batch_images(image_paths)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Batch processing failed"))
        
        # Create results archive
        archive_path = f"/tmp/batch_results_{uuid.uuid4().hex}.zip"
        import zipfile
        
        with zipfile.ZipFile(archive_path, 'w') as zipf:
            for batch_result in result["batch_results"]:
                if "annotated_path" in batch_result:
                    zipf.write(batch_result["annotated_path"], f"annotated_{batch_result['index']}.jpg")
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_dir, batch_dir)
        
        # Create response
        batch_response = BatchProcessingResult(
            success=True,
            total_images=result["summary"]["total_images"],
            processed_images=result["summary"]["processed_images"],
            failed_images=result["summary"]["failed_images"],
            total_detections=result["summary"]["total_detections"],
            processing_time=result["summary"]["processing_time"],
            batch_results=[{
                **res, 
                "annotated_path": f"/batch_results/{Path(res['annotated_path']).name}" if "annotated_path" in res else None,
                "original_path": f"/batch_results/{Path(res['original_path']).name}" if "original_path" in res else None
            } for res in result["batch_results"]],
            results_archive=Path(archive_path).name
        )
        
        return batch_response
        
    except Exception as e:
        logging.error(f"Error in batch processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/download/batch/{archive_name}")
async def download_batch_results(archive_name: str):
    """
    Download batch processing results archive
    """
    try:
        file_path = Path(f"/tmp/{archive_name}")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Archive not found")
        
        return FileResponse(
            path=str(file_path),
            filename=f"deplyze_studio_batch_results.zip",
            media_type="application/zip"
        )
        
    except Exception as e:
        logging.error(f"Error downloading batch results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def cleanup_file(file_path: str):
    """Background task to clean up a file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Cleaned up file: {file_path}")
    except Exception as e:
        logging.warning(f"Failed to cleanup file {file_path}: {e}")

@api_router.post("/detect/video", response_model=VideoProcessingResult)
async def process_video_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Process uploaded video file with logo detection
    """
    try:
        # Validate file type
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        # Create temporary directories
        temp_dir = Path(tempfile.mkdtemp())
        input_path = temp_dir / f"input_{uuid.uuid4().hex[:8]}.mp4"
        output_path = temp_dir / f"output_{uuid.uuid4().hex[:8]}.mp4"
        
        # Save uploaded file
        with open(input_path, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)
        
        logging.info(f"Processing video file: {file.filename}, Size: {len(contents)} bytes")
        
        # Process video
        result = ml_service.process_video_file(str(input_path), str(output_path))
        
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error", "Video processing failed"))
        
        # Move output to permanent location
        output_dir = Path("/tmp/processed_videos")
        output_dir.mkdir(exist_ok=True)
        
        final_output_name = f"processed_{uuid.uuid4().hex[:8]}.mp4"
        final_output_path = output_dir / final_output_name
        
        shutil.move(str(output_path), str(final_output_path))
        
        # Schedule cleanup of temp directory
        background_tasks.add_task(cleanup_temp_dir, temp_dir)
        
        # Create response
        video_result = VideoProcessingResult(
            success=result["success"],
            processed_frames=result["processed_frames"],
            total_detections=result["total_detections"],
            processing_time=result["processing_time"],
            fps=result["fps"],
            resolution=result["resolution"],
            avg_detections_per_frame=result["avg_detections_per_frame"],
            output_filename=final_output_name,
            detected_classes=result.get("detected_classes", [])
        )
        
        # Save to database
        try:
            await db.video_processing.insert_one(video_result.dict())
        except Exception as e:
            logging.warning(f"Failed to save video processing result: {e}")
        
        return video_result
        
    except Exception as e:
        logging.error(f"Error in video processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/track", response_model=TrackingProcessingResult)
async def track_video(payload: TrackRequest, background_tasks: BackgroundTasks):
    try:
        temp_dir = Path(tempfile.mkdtemp())
        input_path = temp_dir / f"input_{uuid.uuid4().hex[:8]}.mp4"
        output_path = temp_dir / f"tracked_{uuid.uuid4().hex[:8]}.mp4"
        input_path.write_bytes(base64.b64decode(payload.video))

        result = ml_service.track_video_file(
            str(input_path),
            str(output_path),
            model_name=payload.model,
            tracker=payload.tracker,
        )
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error", "Tracking failed"))

        output_dir = Path("/tmp/processed_videos")
        output_dir.mkdir(exist_ok=True)
        final_output_name = f"tracked_{uuid.uuid4().hex[:8]}.mp4"
        final_output_path = output_dir / final_output_name
        shutil.move(str(output_path), str(final_output_path))
        background_tasks.add_task(cleanup_temp_dir, temp_dir)

        return TrackingProcessingResult(
            success=True,
            processed_frames=result["processed_frames"],
            total_detections=result["total_detections"],
            processing_time=result["processing_time"],
            fps=result["fps"],
            resolution=result["resolution"],
            avg_detections_per_frame=result["avg_detections_per_frame"],
            output_filename=final_output_name,
            tracker=result["tracker"],
            model_used=result["model_used"],
            total_tracks=result["total_tracks"],
            frame_results=result["frame_results"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error tracking video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/download/video/{filename}")
async def download_processed_video(filename: str):
    """
    Download processed video file
    """
    try:
        file_path = Path("/tmp/processed_videos") / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=str(file_path),
            filename=f"logo_detection_{filename}",
            media_type="video/mp4"
        )
        
    except Exception as e:
        logging.error(f"Error downloading video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/video/history")
async def get_video_processing_history(limit: int = 10):
    """Get recent video processing history"""
    try:
        history = await db.video_processing.find().sort("timestamp", -1).limit(limit).to_list(length=None)
        return history
    except Exception as e:
        logging.error(f"Error getting video history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def cleanup_temp_dir(temp_dir: Path):
    """Background task to clean up temporary directory"""
    try:
        shutil.rmtree(temp_dir)
        logging.info(f"Cleaned up temp directory: {temp_dir}")
    except Exception as e:
        logging.warning(f"Failed to cleanup temp directory {temp_dir}: {e}")

@api_router.post("/detect/image", response_model=DetectionResult)
async def detect_logos_in_image(file: UploadFile = File(...)):
    """
    Detect logos in uploaded image
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await file.read()
        pil_image = Image.open(BytesIO(contents))
        
        # Convert to OpenCV format
        cv_image = pil_to_opencv(pil_image)
        
        # Run detection
        result = ml_service.detect_logos_in_image(cv_image)
        
        # Create response
        detection_result = DetectionResult(
            detections=[Detection(**det) for det in result["detections"]],
            inference_time=result["inference_time"],
            original_shape=list(result["original_shape"]),
            model_input_size=result["model_input_size"],
            error=result.get("error")
        )
        
        # Save annotated image to file system for download
        try:
            annotated_dir = Path("/tmp/annotated_images")
            annotated_dir.mkdir(exist_ok=True)
            
            annotated_path = annotated_dir / f"{detection_result.id}.jpg"
            cv2.imwrite(str(annotated_path), result["annotated_image"])
            
            # Save to database with image path
            history_entry = DetectionHistory(
                id=detection_result.id,
                detection_result=detection_result,
                image_name=file.filename or "unknown.jpg"
            )
            await db.detection_history.insert_one(history_entry.dict())
        except Exception as e:
            logging.warning(f"Failed to save detection history: {e}")
        
        return detection_result
        
    except Exception as e:
        logging.error(f"Error in image detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/detect/image/annotated")
async def detect_logos_with_annotation(file: UploadFile = File(...)):
    """
    Detect logos and return annotated image
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await file.read()
        pil_image = Image.open(BytesIO(contents))
        
        # Convert to OpenCV format
        cv_image = pil_to_opencv(pil_image)
        
        # Run detection
        result = ml_service.detect_logos_in_image(cv_image)
        
        # Convert annotated image to base64
        annotated_base64 = numpy_to_base64(result["annotated_image"])
        
        return {
            "detections": result["detections"],
            "inference_time": result["inference_time"],
            "annotated_image": annotated_base64,
            "original_shape": result["original_shape"]
        }
        
    except Exception as e:
        logging.error(f"Error in image annotation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/history", response_model=List[DetectionHistory])
async def get_detection_history(limit: int = 10):
    """Get recent detection history"""
    try:
        history = await db.detection_history.find().sort("created_at", -1).limit(limit).to_list(length=None)
        return [DetectionHistory(**item) for item in history]
    except Exception as e:
        logging.error(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket for real-time video detection
class VideoConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def send_detection_result(self, websocket: WebSocket, data: dict):
        try:
            await websocket.send_text(json.dumps(data))
        except Exception as e:
            logging.error(f"Error sending WebSocket data: {e}")
            self.disconnect(websocket)

video_manager = VideoConnectionManager()

@api_router.websocket("/detect/video")
async def websocket_video_detection(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video logo detection
    """
    await video_manager.connect(websocket)
    
    try:
        while True:
            # Receive frame data (base64 encoded)
            data = await websocket.receive_text()
            frame_data = json.loads(data)
            
            if "frame" in frame_data:
                # Decode frame
                frame = base64_to_numpy(frame_data["frame"])
                
                # Run detection (optimized for video)
                result = ml_service.detect_logos_in_frame(frame)
                
                # Send results back
                response = {
                    "type": "detection_result",
                    "detections": result["detections"],
                    "inference_time": result["inference_time"],
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                await video_manager.send_detection_result(websocket, response)
                
    except WebSocketDisconnect:
        video_manager.disconnect(websocket)
        logging.info("WebSocket client disconnected")
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
        video_manager.disconnect(websocket)

# Original status check endpoints
@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

from fastapi.responses import Response, FileResponse, StreamingResponse

@api_router.get("/proxy/model")
async def proxy_model(url: str):
    """
    Proxy model downloads to bypass CORS restrictions for GitHub etc.
    """
    import requests
    try:
        def generate():
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=8192):
                    yield chunk
        
        return StreamingResponse(
            generate(),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={url.split('/')[-1]}",
            }
        )
    except Exception as e:
        logging.error(f"Error proxying model {url}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files for batch results
batch_res_path = Path("/tmp/batch_results")
batch_res_path.mkdir(exist_ok=True, parents=True)
app.mount("/batch_results", StaticFiles(directory=str(batch_res_path)), name="batch_results")

# Mount static files for processed videos
processed_videos_path = Path("/tmp/processed_videos")
processed_videos_path.mkdir(exist_ok=True, parents=True)
app.mount("/processed_videos", StaticFiles(directory=str(processed_videos_path)), name="processed_videos")

# Include the router in the main app
app.include_router(api_router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=get_allowed_origins(),
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Deplyze Studio API...")
    logger.info(f"Model device: {ml_service.device}")
    logger.info("API is ready!")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
