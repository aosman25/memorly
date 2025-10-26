import logging
import os
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

from config import Config
from models import (
    ProcessMediaRequest,
    ProcessMediaResponse,
    HealthResponse,
    MediaType
)
from service_clients import ServiceClients
from mongodb_client import MongoDBClient
from pipelines import MediaPipeline

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances
service_clients = None
mongo_client = None
media_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    global service_clients, mongo_client, media_pipeline

    # Startup
    logger.info("Starting Gateway Service...")

    # Initialize clients
    service_clients = ServiceClients()
    mongo_client = MongoDBClient(Config.MONGO_URI, Config.MONGO_DB_NAME)
    media_pipeline = MediaPipeline(service_clients, mongo_client)

    # Create upload directory
    os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
    logger.info(f"Upload directory: {Config.UPLOAD_DIR}")

    logger.info("Gateway Service started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Gateway Service...")
    if mongo_client:
        mongo_client.close()
    logger.info("Gateway Service shutdown complete")


app = FastAPI(
    title="Memorly Gateway Service",
    description="Gateway service for processing images, videos, and text",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    services = {
        "extract-features": await service_clients.health_check(Config.EXTRACT_FEATURES_SERVICE_URL),
        "face-extraction": await service_clients.health_check(Config.FACE_EXTRACTION_SERVICE_URL),
        "embed": await service_clients.health_check(Config.EMBED_SERVICE_URL),
        "upsert": await service_clients.health_check(Config.UPSERT_SERVICE_URL),
        "video-segmentation": await service_clients.health_check(Config.VIDEO_SEGMENTATION_SERVICE_URL),
        "mongodb": mongo_client is not None
    }

    all_healthy = all(services.values())
    status = "healthy" if all_healthy else "degraded"

    return HealthResponse(status=status, services=services)


@app.post("/process/image", response_model=ProcessMediaResponse)
async def process_image(
    file: UploadFile = File(...),
    media_id: str = Form(...),
    user_id: str = Form(...),
    timestamp: int = Form(...),
    location: str = Form(None)
):
    """
    Process an image through the full pipeline.

    Pipeline:
    1. Extract features (objects, content, tags)
    2. Extract and process faces
    3. Generate embedding
    4. Upsert to vector database
    """
    logger.info(f"Received image processing request: media_id={media_id}, user_id={user_id}")

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Save uploaded file
    file_path = os.path.join(Config.UPLOAD_DIR, f"{media_id}_{file.filename}")
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process through pipeline
        result = await media_pipeline.process_image(
            user_id=user_id,
            media_id=media_id,
            media_path=file_path,
            timestamp=timestamp,
            location=location
        )

        return ProcessMediaResponse(**result)

    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

    finally:
        # Cleanup uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Cleaned up temporary file: {file_path}")


@app.post("/process/video", response_model=ProcessMediaResponse)
async def process_video(
    file: UploadFile = File(...),
    media_id: str = Form(...),
    user_id: str = Form(...),
    timestamp: int = Form(...),
    location: str = Form(None)
):
    """
    Process a video through the full pipeline.

    Pipeline:
    1. Segment video into scenes
    2. Extract features from representative frames
    3. Extract and process faces
    4. Generate fused embedding (visual + text)
    5. Upsert to vector database
    """
    logger.info(f"Received video processing request: media_id={media_id}, user_id={user_id}")

    # Validate file type
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    # Save uploaded file
    file_path = os.path.join(Config.UPLOAD_DIR, f"{media_id}_{file.filename}")
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process through pipeline
        result = await media_pipeline.process_video(
            user_id=user_id,
            media_id=media_id,
            media_path=file_path,
            timestamp=timestamp,
            location=location
        )

        return ProcessMediaResponse(**result)

    except Exception as e:
        logger.error(f"Error processing video: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

    finally:
        # Cleanup uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Cleaned up temporary file: {file_path}")


@app.post("/process/text", response_model=ProcessMediaResponse)
async def process_text(
    text: str = Form(...),
    media_id: str = Form(...),
    user_id: str = Form(...),
    timestamp: int = Form(...),
    location: str = Form(None)
):
    """
    Process text through the pipeline.

    Pipeline:
    1. Generate text embedding
    2. Upsert to vector database
    """
    logger.info(f"Received text processing request: media_id={media_id}, user_id={user_id}")

    try:
        # Process through pipeline
        result = await media_pipeline.process_text(
            user_id=user_id,
            media_id=media_id,
            text_content=text,
            timestamp=timestamp,
            location=location
        )

        return ProcessMediaResponse(**result)

    except Exception as e:
        logger.error(f"Error processing text: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=True
    )
