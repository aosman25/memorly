import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
from dotenv import load_dotenv

from utils import (
    embed_image,
    embed_text,
    embed_video_frames,
    normalize_embedding,
)
from models import (
    ImageEmbedRequest,
    TextEmbedRequest,
    VideoEmbedRequest,
    BatchImageEmbedRequest,
    EmbedResponse,
    BatchEmbedResponse,
    HealthResponse,
    ErrorResponse,
)

# Load environment variables
load_dotenv()


# Configuration
class Config:
    DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
    DEEPINFRA_BASE_URL = os.getenv("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/clip-ViT-B-32-multilingual-v1")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "180"))
    PORT = int(os.getenv("PORT", "5004"))
    MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "100"))

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.DEEPINFRA_API_KEY:
            raise ValueError("DEEPINFRA_API_KEY environment variable is required")


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
openai_client = None
logger = structlog.get_logger()


# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global openai_client

    # Startup
    try:
        logger.info("Starting Embedding Service")
        Config.validate()

        # Initialize OpenAI client
        openai_client = OpenAI(
            api_key=Config.DEEPINFRA_API_KEY,
            base_url=Config.DEEPINFRA_BASE_URL,
        )

        logger.info(
            "Service started successfully",
            model=Config.EMBEDDING_MODEL
        )
        yield

    except Exception as e:
        logger.error("Failed to start service", error=str(e))
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Embedding Service")
        openai_client = None


# FastAPI app
app = FastAPI(
    title="Embedding API",
    description="Generate embeddings for images and videos using CLIP multilingual model",
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


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        model=Config.EMBEDDING_MODEL
    )


@app.get("/ready", response_model=HealthResponse)
async def readiness_check():
    """Readiness check endpoint"""
    try:
        if openai_client is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not ready - OpenAI client not initialized",
            )

        return HealthResponse(
            status="ready",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            model=Config.EMBEDDING_MODEL
        )
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready",
        )


@app.post(
    "/embed/text",
    response_model=EmbedResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        408: {"model": ErrorResponse, "description": "Request timeout"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
)
async def embed_single_text(request: TextEmbedRequest, http_request: Request):
    """
    Generate embedding for plain text.

    Returns a 512-dimensional CLIP embedding vector.
    """
    request_id = http_request.headers.get(
        "x-request-id", f"req_{int(time.time() * 1000)}"
    )
    start_time = time.time()

    try:
        if openai_client is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service unavailable - OpenAI client not initialized",
            )

        logger.info("Embedding text", text_length=len(request.text), request_id=request_id)

        # Generate embedding
        embedding = await asyncio.to_thread(
            embed_text,
            request.text,
            openai_client,
            Config.EMBEDDING_MODEL
        )

        processing_time_ms = (time.time() - start_time) * 1000

        logger.info(
            "Text embedding completed",
            dimension=len(embedding),
            processing_time_ms=processing_time_ms,
            request_id=request_id
        )

        return EmbedResponse(
            embedding=embedding.tolist(),
            request_id=request_id,
            processing_time_ms=processing_time_ms
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Text embedding failed", error=str(e), request_id=request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate text embedding",
        )


@app.post(
    "/embed/image",
    response_model=EmbedResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        408: {"model": ErrorResponse, "description": "Request timeout"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
)
async def embed_single_image(request: ImageEmbedRequest, http_request: Request):
    """
    Generate embedding for a single image.

    Returns a 512-dimensional CLIP embedding vector.
    """
    request_id = http_request.headers.get(
        "x-request-id", f"req_{int(time.time() * 1000)}"
    )
    start_time = time.time()

    try:
        if openai_client is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service unavailable - OpenAI client not initialized",
            )

        logger.info("Embedding single image", request_id=request_id)

        # Generate embedding
        embedding = await asyncio.to_thread(
            embed_image,
            request.image_base64,
            openai_client,
            Config.EMBEDDING_MODEL
        )

        processing_time_ms = (time.time() - start_time) * 1000

        logger.info(
            "Image embedding completed",
            dimension=len(embedding),
            processing_time_ms=processing_time_ms,
            request_id=request_id
        )

        return EmbedResponse(
            embedding=embedding.tolist(),
            request_id=request_id,
            processing_time_ms=processing_time_ms
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Image embedding failed", error=str(e), request_id=request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate embedding",
        )


@app.post(
    "/embed/video",
    response_model=EmbedResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        408: {"model": ErrorResponse, "description": "Request timeout"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
)
async def embed_video(request: VideoEmbedRequest, http_request: Request):
    """
    Generate fused embedding for video frames.

    For each frame:
    - Generates visual embedding from the frame image
    - Generates text embedding from transcript (if available)
    - Fuses embeddings using: alpha * visual + beta * text
    - Default weights: alpha=0.6 (visual), beta=0.4 (text)

    Returns a single 512-dimensional embedding representing the entire video.
    """
    request_id = http_request.headers.get(
        "x-request-id", f"req_{int(time.time() * 1000)}"
    )
    start_time = time.time()

    try:
        if openai_client is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service unavailable - OpenAI client not initialized",
            )

        logger.info(
            "Embedding video",
            frame_count=len(request.frames),
            visual_weight=request.visual_weight,
            text_weight=request.text_weight,
            request_id=request_id
        )

        # Prepare frames
        frames = [
            (frame.frame_base64, frame.transcript)
            for frame in request.frames
        ]

        # Generate fused embedding
        embedding = await asyncio.to_thread(
            embed_video_frames,
            frames,
            openai_client,
            request.visual_weight,
            request.text_weight,
            Config.EMBEDDING_MODEL
        )

        processing_time_ms = (time.time() - start_time) * 1000

        logger.info(
            "Video embedding completed",
            dimension=len(embedding),
            processing_time_ms=processing_time_ms,
            request_id=request_id
        )

        return EmbedResponse(
            embedding=embedding.tolist(),
            request_id=request_id,
            processing_time_ms=processing_time_ms
        )

    except asyncio.TimeoutError:
        logger.error("Video embedding timeout", request_id=request_id)
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="Video embedding timeout",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Video embedding failed", error=str(e), request_id=request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate video embedding",
        )


@app.post(
    "/embed/batch",
    response_model=BatchEmbedResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        408: {"model": ErrorResponse, "description": "Request timeout"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
)
async def embed_batch_images(request: BatchImageEmbedRequest, http_request: Request):
    """
    Generate embeddings for multiple images in batch.

    Processes up to 100 images and returns their embeddings.
    """
    request_id = http_request.headers.get(
        "x-request-id", f"req_{int(time.time() * 1000)}"
    )
    start_time = time.time()

    try:
        if openai_client is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service unavailable - OpenAI client not initialized",
            )

        if len(request.images) > Config.MAX_BATCH_SIZE:
            raise ValueError(f"Batch size exceeds maximum of {Config.MAX_BATCH_SIZE}")

        logger.info(
            "Embedding batch",
            image_count=len(request.images),
            request_id=request_id
        )

        # Generate embeddings for all images
        tasks = [
            asyncio.to_thread(
                embed_image,
                image_base64,
                openai_client,
                Config.EMBEDDING_MODEL
            )
            for image_base64 in request.images
        ]

        embeddings = await asyncio.gather(*tasks)

        processing_time_ms = (time.time() - start_time) * 1000

        logger.info(
            "Batch embedding completed",
            count=len(embeddings),
            processing_time_ms=processing_time_ms,
            request_id=request_id
        )

        return BatchEmbedResponse(
            embeddings=[emb.tolist() for emb in embeddings],
            count=len(embeddings),
            request_id=request_id,
            processing_time_ms=processing_time_ms
        )

    except asyncio.TimeoutError:
        logger.error("Batch embedding timeout", request_id=request_id)
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="Batch embedding timeout",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Batch embedding failed", error=str(e), request_id=request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate batch embeddings",
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
