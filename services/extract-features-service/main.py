import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import List

import structlog
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from dotenv import load_dotenv

from utils import (
    load_image_from_url,
    load_image_from_base64,
    create_extraction_prompt,
)
from models import (
    ImageFeatures,
    ExtractionRequest,
    BatchExtractionRequest,
    ExtractionResponse,
    BatchExtractionResponse,
    HealthResponse,
    ErrorResponse,
)

# Load environment variables
load_dotenv()


# Configuration
class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    MAX_IMAGES_PER_REQUEST = int(os.getenv("MAX_IMAGES_PER_REQUEST", "10"))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    PORT = int(os.getenv("PORT", "5001"))

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is required")


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
gemini_client = None
logger = structlog.get_logger()


# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global gemini_client

    # Startup
    try:
        logger.info("Starting Feature Extraction Service")
        Config.validate()

        # Initialize Gemini client
        gemini_client = genai.Client(api_key=Config.GEMINI_API_KEY)

        # Test client connection
        await test_gemini_connection()

        logger.info("Service started successfully", model=Config.GEMINI_MODEL)
        yield

    except Exception as e:
        logger.error("Failed to start service", error=str(e))
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Feature Extraction Service")
        gemini_client = None


async def test_gemini_connection():
    """Test Gemini API connection during startup"""
    try:
        response = gemini_client.models.generate_content(
            model=Config.GEMINI_MODEL,
            contents="test",
            config=types.GenerateContentConfig(response_mime_type="text/plain"),
        )
        logger.info("Gemini API connection verified")
    except Exception as e:
        logger.error("Failed to connect to Gemini API", error=str(e))
        raise


# FastAPI app
app = FastAPI(
    title="Image Feature Extraction API",
    description="Extract objects, content descriptions, and tags from images using Gemini AI",
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


# Retry logic for Gemini API calls
@retry(
    retry=retry_if_exception_type((Exception,)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
async def call_gemini_api(image_part: types.Part) -> ImageFeatures:
    """Call Gemini API with retry logic"""
    try:
        prompt_parts = create_extraction_prompt()

        response = gemini_client.models.generate_content(
            model=Config.GEMINI_MODEL,
            contents=[*prompt_parts, image_part],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ImageFeatures,
            ),
        )

        if not response.parsed:
            logger.warning("Empty response from Gemini API")
            return ImageFeatures(objects=[], content="No description available", tags=[])

        return response.parsed

    except Exception as e:
        logger.error("Gemini API call failed", error=str(e))
        raise


async def process_single_image(
    extraction_request: ExtractionRequest, request_id: str
) -> ExtractionResponse:
    """Process a single image with proper error handling"""
    start_time = time.time()

    try:
        logger.info("Processing image", request_id=request_id)

        # Load image from URL or base64
        if extraction_request.image_url:
            logger.info("Loading image from URL", url=extraction_request.image_url)
            image_part = load_image_from_url(extraction_request.image_url)
        elif extraction_request.image_base64:
            logger.info("Loading image from base64")
            image_part = load_image_from_base64(extraction_request.image_base64)
        else:
            raise ValueError("No image source provided")

        # Extract features with timeout
        features = await asyncio.wait_for(
            call_gemini_api(image_part), timeout=Config.REQUEST_TIMEOUT
        )

        processing_time_ms = (time.time() - start_time) * 1000

        logger.info(
            "Image processed successfully",
            processing_time_ms=processing_time_ms,
            objects_count=len(features.objects),
            tags_count=len(features.tags),
            request_id=request_id,
        )

        return ExtractionResponse(
            features=features,
            request_id=request_id,
            processing_time_ms=processing_time_ms,
        )

    except asyncio.TimeoutError:
        logger.error("Image processing timeout", request_id=request_id)
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="Image processing timeout",
        )
    except Exception as e:
        logger.error("Image processing failed", error=str(e), request_id=request_id)
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        gemini_model=Config.GEMINI_MODEL,
    )


@app.get("/ready", response_model=HealthResponse)
async def readiness_check():
    """Readiness check endpoint"""
    try:
        if gemini_client is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not ready - Gemini client not initialized",
            )

        return HealthResponse(
            status="ready",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            gemini_model=Config.GEMINI_MODEL,
        )
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready",
        )


@app.post(
    "/extract",
    response_model=ExtractionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        408: {"model": ErrorResponse, "description": "Request timeout"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
)
async def extract_features(request: ExtractionRequest, http_request: Request):
    """
    Extract features from a single image.

    Processes the image and returns:
    - Objects: List of detected objects
    - Content: Descriptive caption
    - Tags: Relevant tags and categories
    """
    request_id = http_request.headers.get(
        "x-request-id", f"req_{int(time.time() * 1000)}"
    )

    try:
        if gemini_client is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service unavailable - Gemini client not initialized",
            )

        result = await process_single_image(request, request_id)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Feature extraction failed", error=str(e), request_id=request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to extract features",
        )


@app.post(
    "/extract-batch",
    response_model=BatchExtractionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
)
async def extract_features_batch(request: BatchExtractionRequest, http_request: Request):
    """
    Extract features from multiple images concurrently.

    Processes up to 10 images in parallel with proper error handling,
    timeouts, and retry logic.
    """
    request_id = http_request.headers.get(
        "x-request-id", f"req_{int(time.time() * 1000)}"
    )
    start_time = time.time()

    try:
        if gemini_client is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service unavailable - Gemini client not initialized",
            )

        # Process all images concurrently
        tasks = [
            process_single_image(img_request, f"{request_id}_{i}")
            for i, img_request in enumerate(request.images)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and collect successful results
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Image processing failed in batch",
                    index=i,
                    error=str(result),
                    request_id=request_id,
                )
            else:
                successful_results.append(result)

        total_processing_time_ms = (time.time() - start_time) * 1000

        logger.info(
            "Batch processing completed",
            input_count=len(request.images),
            success_count=len(successful_results),
            total_time_ms=total_processing_time_ms,
            request_id=request_id,
        )

        return BatchExtractionResponse(
            results=successful_results,
            processed_count=len(successful_results),
            request_id=request_id,
            total_processing_time_ms=total_processing_time_ms,
        )

    except Exception as e:
        logger.error("Batch processing failed", error=str(e), request_id=request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process batch",
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
