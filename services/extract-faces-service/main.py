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
from dotenv import load_dotenv

from utils import (
    load_image_from_url,
    load_image_from_base64,
    process_image_for_faces,
    deduplicate_faces,
    face_image_to_base64,
    check_deepface_models,
)
from models import (
    Face,
    ImageInput,
    ExtractionRequest,
    ExtractionResponse,
    HealthResponse,
    ErrorResponse,
)

# Load environment variables
load_dotenv()


# Configuration
class Config:
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "180"))  # 3 minutes
    PORT = int(os.getenv("PORT", "5003"))
    MAX_IMAGES_PER_REQUEST = int(os.getenv("MAX_IMAGES_PER_REQUEST", "100"))
    MIN_FACE_SIZE = int(os.getenv("MIN_FACE_SIZE", "20"))
    FACE_MODEL = os.getenv("FACE_MODEL", "ArcFace")
    DETECTOR_BACKEND = os.getenv("DETECTOR_BACKEND", "retinaface")

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        # Check if DeepFace models are available
        if not check_deepface_models():
            logger.warning("DeepFace models not yet loaded - they will be downloaded on first use")


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
        logger.info("Starting Face Extraction Service")
        Config.validate()
        logger.info(
            "Service started successfully",
            model=Config.FACE_MODEL,
            detector=Config.DETECTOR_BACKEND
        )
        yield

    except Exception as e:
        logger.error("Failed to start service", error=str(e))
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Face Extraction Service")


# FastAPI app
app = FastAPI(
    title="Face Extraction API",
    description="Extract and deduplicate faces from images using ArcFace embeddings",
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


async def process_extraction_request(
    request: ExtractionRequest,
    request_id: str
) -> ExtractionResponse:
    """
    Process face extraction request: detect faces, generate embeddings, deduplicate.

    Args:
        request: Extraction request
        request_id: Request identifier

    Returns:
        Extraction response with unique faces
    """
    start_time = time.time()

    try:
        logger.info(
            "Processing extraction request",
            image_count=len(request.images),
            similarity_threshold=request.similarity_threshold,
            request_id=request_id
        )

        all_faces = []
        images_processed = 0

        # Process each image
        for i, image_input in enumerate(request.images, 1):
            try:
                logger.info(f"Processing image {i}/{len(request.images)}")

                # Load image
                if image_input.url:
                    image = await asyncio.to_thread(
                        load_image_from_url, image_input.url
                    )
                elif image_input.base64:
                    image = await asyncio.to_thread(
                        load_image_from_base64, image_input.base64
                    )
                else:
                    logger.warning(f"Skipping image {i}: no source provided")
                    continue

                # Process image for faces
                faces = await asyncio.to_thread(
                    process_image_for_faces,
                    image,
                    Config.MIN_FACE_SIZE,
                    Config.FACE_MODEL,
                    Config.DETECTOR_BACKEND
                )

                all_faces.extend(faces)
                images_processed += 1

                logger.info(
                    f"Image {i} processed",
                    faces_detected=len(faces),
                    total_faces_so_far=len(all_faces)
                )

            except Exception as e:
                logger.error(f"Failed to process image {i}", error=str(e))
                continue

        total_faces_detected = len(all_faces)

        # Deduplicate faces
        logger.info("Deduplicating faces", total_faces=total_faces_detected)

        unique_faces_data = await asyncio.to_thread(
            deduplicate_faces,
            all_faces,
            request.similarity_threshold
        )

        # Convert to response format
        faces = []
        for face_image, embedding, confidence, facial_area in unique_faces_data:
            # Convert face image to base64
            face_image_base64 = await asyncio.to_thread(
                face_image_to_base64, face_image
            )

            faces.append(Face(
                face_image=face_image_base64,
                embedding=embedding.tolist(),
                confidence=confidence,
                facial_area=facial_area
            ))

        processing_time_ms = (time.time() - start_time) * 1000

        logger.info(
            "Extraction completed",
            images_processed=images_processed,
            total_faces_detected=total_faces_detected,
            unique_faces=len(faces),
            processing_time_ms=processing_time_ms,
            request_id=request_id
        )

        return ExtractionResponse(
            faces=faces,
            total_faces_detected=total_faces_detected,
            unique_faces=len(faces),
            images_processed=images_processed,
            request_id=request_id,
            processing_time_ms=processing_time_ms
        )

    except Exception as e:
        logger.error("Face extraction failed", error=str(e), request_id=request_id)
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        deepface_loaded=True
    )


@app.get("/ready", response_model=HealthResponse)
async def readiness_check():
    """Readiness check endpoint"""
    try:
        return HealthResponse(
            status="ready",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            face_detection_model=f"{Config.FACE_MODEL} + {Config.DETECTOR_BACKEND}"
        )
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready",
        )


@app.post(
    "/extract-faces",
    response_model=ExtractionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        408: {"model": ErrorResponse, "description": "Request timeout"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
)
async def extract_faces(request: ExtractionRequest, http_request: Request):
    """
    Extract unique faces from a list of images.

    This endpoint:
    1. Detects faces in all provided images
    2. Generates ArcFace embeddings (512-dimensional) for each face
    3. Deduplicates faces using cosine similarity (threshold: 0.4)
    4. Returns unique faces with embeddings and headshot images

    **Note**: Faces with cosine similarity >= 0.4 are considered the same person.
    """
    request_id = http_request.headers.get(
        "x-request-id", f"req_{int(time.time() * 1000)}"
    )

    try:
        # Validate request
        if len(request.images) > Config.MAX_IMAGES_PER_REQUEST:
            raise ValueError(
                f"Too many images. Maximum allowed: {Config.MAX_IMAGES_PER_REQUEST}"
            )

        # Process with timeout
        result = await asyncio.wait_for(
            process_extraction_request(request, request_id),
            timeout=Config.REQUEST_TIMEOUT
        )
        return result

    except asyncio.TimeoutError:
        logger.error("Face extraction timeout", request_id=request_id)
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="Face extraction timeout",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Face extraction failed", error=str(e), request_id=request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to extract faces",
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
