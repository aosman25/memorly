import asyncio
import logging
import os
import sys
import time
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from dotenv import load_dotenv

from utils import (
    download_video,
    save_base64_video,
    detect_scenes,
    extract_frame,
    frame_to_base64,
    extract_audio_segment,
    transcribe_audio,
    get_video_duration,
    check_ffmpeg_available,
    check_ffprobe_available,
)
from models import (
    VideoScene,
    SegmentationRequest,
    SegmentationResponse,
    HealthResponse,
    ErrorResponse,
)

# Load environment variables
load_dotenv()


# Configuration
class Config:
    WHISPER_API_KEY = os.getenv("WHISPER_API_KEY")
    WHISPER_BASE_URL = os.getenv("WHISPER_BASE_URL", "https://api.deepinfra.com/v1/openai")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))  # 5 minutes for video processing
    PORT = int(os.getenv("PORT", "5002"))
    MAX_VIDEO_SIZE_MB = int(os.getenv("MAX_VIDEO_SIZE_MB", "500"))

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.WHISPER_API_KEY:
            raise ValueError("WHISPER_API_KEY environment variable is required")

        if not check_ffmpeg_available():
            raise RuntimeError("ffmpeg is not installed or not available in PATH")

        if not check_ffprobe_available():
            raise RuntimeError("ffprobe is not installed or not available in PATH")


# Logging setup
def setup_logging():
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, Config.LOG_LEVEL),
    )


# Global variables
logger = structlog.get_logger()


# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    try:
        logger.info("Starting Video Segmentation Service")
        Config.validate()
        logger.info("Service started successfully")
        yield

    except Exception as e:
        logger.error("Failed to start service", error=str(e))
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Video Segmentation Service")


# FastAPI app
app = FastAPI(
    title="Video Segmentation API",
    description="Segment videos by scene changes and generate transcripts using ffmpeg and Whisper",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=(
        os.getenv("ALLOWED_ORIGINS", "").split(",")
        if os.getenv("ALLOWED_ORIGINS")
        else ["*"]
    ),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Request/response logging and timing"""
    request_id = request.headers.get("x-request-id", f"req_{int(time.time() * 1000)}")
    start_time = time.time()

    logger.info(
        "Request started",
        method=request.method,
        path=request.url.path,
        request_id=request_id,
    )

    response = await call_next(request)

    duration = time.time() - start_time
    logger.info(
        "Request completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration=f"{duration:.3f}s",
        request_id=request_id,
    )

    response.headers["x-request-id"] = request_id
    return response


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    request_id = request.headers.get("x-request-id", "unknown")
    logger.error("Validation error", error=str(exc), request_id=request_id)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error=str(exc),
            request_id=request_id,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    request_id = request.headers.get("x-request-id", "unknown")
    logger.error("Unhandled exception", error=str(exc), request_id=request_id)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            request_id=request_id,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        ).model_dump(),
    )


async def process_video(request: SegmentationRequest, request_id: str) -> SegmentationResponse:
    """
    Process video: detect scenes, extract frames, and transcribe audio.

    Args:
        request: Segmentation request
        request_id: Request identifier

    Returns:
        Segmentation response with scenes
    """
    start_time = time.time()
    temp_dir = None

    try:
        # Create temporary directory for processing
        temp_dir = tempfile.mkdtemp(prefix="video_seg_")
        logger.info("Created temp directory", path=temp_dir, request_id=request_id)

        # Download or save video
        video_path = os.path.join(temp_dir, "input_video.mp4")

        if request.video_url:
            logger.info("Downloading video from URL", url=request.video_url)
            await asyncio.to_thread(download_video, request.video_url, video_path)
        elif request.video_base64:
            logger.info("Saving video from base64")
            await asyncio.to_thread(save_base64_video, request.video_base64, video_path)

        # Check video size
        video_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        if video_size_mb > Config.MAX_VIDEO_SIZE_MB:
            raise ValueError(f"Video size ({video_size_mb:.1f}MB) exceeds maximum allowed ({Config.MAX_VIDEO_SIZE_MB}MB)")

        logger.info("Video ready for processing", size_mb=f"{video_size_mb:.1f}")

        # Get video duration
        duration = await asyncio.to_thread(get_video_duration, video_path)

        # Detect scenes
        logger.info("Detecting scenes", threshold=request.scene_threshold)
        scene_ranges = await asyncio.to_thread(
            detect_scenes, video_path, request.scene_threshold
        )

        logger.info("Processing scenes", scene_count=len(scene_ranges))

        # Process each scene
        scenes = []
        for i, (start_time_scene, end_time_scene) in enumerate(scene_ranges, 1):
            logger.info(
                "Processing scene",
                scene_num=i,
                start=start_time_scene,
                end=end_time_scene
            )

            # Extract frame (middle of scene)
            frame_timestamp = (start_time_scene + end_time_scene) / 2
            frame_path = os.path.join(temp_dir, f"frame_{i}.jpg")

            await asyncio.to_thread(
                extract_frame, video_path, frame_timestamp, frame_path
            )

            # Convert frame to base64
            frame_base64 = await asyncio.to_thread(frame_to_base64, frame_path)

            # Extract and transcribe audio segment
            audio_path = os.path.join(temp_dir, f"audio_{i}.mp3")

            await asyncio.to_thread(
                extract_audio_segment,
                video_path,
                start_time_scene,
                end_time_scene,
                audio_path
            )

            # Transcribe audio
            transcript = await asyncio.to_thread(
                transcribe_audio,
                audio_path,
                Config.WHISPER_API_KEY,
                Config.WHISPER_BASE_URL,
                request.language
            )

            scenes.append(VideoScene(
                frame_base64=frame_base64,
                transcript=transcript,
                start_timestamp=start_time_scene,
                end_timestamp=end_time_scene,
                scene_number=i
            ))

            logger.info(
                "Scene processed",
                scene_num=i,
                transcript_length=len(transcript)
            )

        processing_time_ms = (time.time() - start_time) * 1000

        logger.info(
            "Video processing completed",
            total_scenes=len(scenes),
            processing_time_ms=processing_time_ms,
            request_id=request_id
        )

        return SegmentationResponse(
            scenes=scenes,
            total_scenes=len(scenes),
            video_duration=duration,
            request_id=request_id,
            processing_time_ms=processing_time_ms
        )

    except Exception as e:
        logger.error("Video processing failed", error=str(e), request_id=request_id)
        raise

    finally:
        # Cleanup temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temp directory", path=temp_dir)
            except Exception as e:
                logger.warning("Failed to cleanup temp directory", error=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        ffmpeg_available=check_ffmpeg_available(),
        whisper_model="whisper-1"
    )


@app.get("/ready", response_model=HealthResponse)
async def readiness_check():
    """Readiness check endpoint"""
    try:
        if not Config.WHISPER_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not ready - Whisper API key not configured",
            )

        if not check_ffmpeg_available() or not check_ffprobe_available():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not ready - ffmpeg/ffprobe not available",
            )

        return HealthResponse(
            status="ready",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            ffmpeg_available=True,
            whisper_model="whisper-1"
        )
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready",
        )


@app.post(
    "/segment",
    response_model=SegmentationResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        408: {"model": ErrorResponse, "description": "Request timeout"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
)
async def segment_video(request: SegmentationRequest, http_request: Request):
    """
    Segment a video into scenes with frame samples and transcripts.

    This endpoint:
    1. Detects scene changes using ffmpeg
    2. Extracts a sample frame from each scene
    3. Transcribes audio for each scene using Whisper
    4. Returns a list of scenes with frames, transcripts, and timestamps
    """
    request_id = http_request.headers.get(
        "x-request-id", f"req_{int(time.time() * 1000)}"
    )

    try:
        # Process video with timeout
        result = await asyncio.wait_for(
            process_video(request, request_id),
            timeout=Config.REQUEST_TIMEOUT
        )
        return result

    except asyncio.TimeoutError:
        logger.error("Video processing timeout", request_id=request_id)
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="Video processing timeout",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Video segmentation failed", error=str(e), request_id=request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to segment video",
        )


if __name__ == "__main__":
    setup_logging()
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=Config.PORT,
        log_config=None,  # Use our custom logging
        access_log=False,  # Handled by middleware
    )
