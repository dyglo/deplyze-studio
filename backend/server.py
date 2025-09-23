from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
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

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Initialize ML service at startup
ml_service = LogoDetectionService()

# Create the main app without a prefix
app = FastAPI(title="VisionFlow API", version="1.0.0")

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

class ModelInfo(BaseModel):
    model_path: str
    device: str
    confidence_threshold: float
    iou_threshold: float
    max_inference_size: int
    bbox_thickness: int
    available_classes: List[str]

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
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
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
    return {"message": "VisionFlow API is running!", "version": "1.0.0"}

@api_router.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get current model information"""
    model_info = ml_service.get_model_info()
    return ModelInfo(**model_info)

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
            filename=f"visionflow_annotated_{image_id}.jpg",
            media_type="image/jpeg"
        )
        
    except Exception as e:
        logging.error(f"Error downloading annotated image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
            output_filename=final_output_name
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

# Include the router in the main app
app.include_router(api_router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
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
    logger.info("Starting VisionFlow API...")
    logger.info(f"Model device: {ml_service.device}")
    logger.info("API is ready!")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()