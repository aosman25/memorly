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
from google import genai
from google.genai import types
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from dotenv import load_dotenv

from utils import create_extraction_prompt, MongoDBClient
from models import (
    QueryFeatures,
    QueryProcessingRequest,
    QueryProcessingResponse,
    HealthResponse,
    ErrorResponse,
)

# Load environment variables
load_dotenv()


# Configuration
class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "memorly")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    PORT = int(os.getenv("PORT", "8000"))

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
mongo_client = None
logger = structlog.get_logger()


# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global gemini_client, mongo_client

    # Startup
    try:
        logger.info("Starting Query Processing Service")
        Config.validate()

        # Initialize Gemini client
        gemini_client = genai.Client(api_key=Config.GEMINI_API_KEY)

        # Initialize MongoDB client
        mongo_client = MongoDBClient(Config.MONGO_URI, Config.MONGO_DB_NAME)

        # Test Gemini connection
        await test_gemini_connection()

        logger.info("Service started successfully", model=Config.GEMINI_MODEL)
        yield

    except Exception as e:
        logger.error("Failed to start service", error=str(e))
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Query Processing Service")
        if mongo_client:
            mongo_client.close()
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
    title="Query Processing API",
    description="Extract structured information from user search queries using Gemini AI",
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
async def call_gemini_api(query: str) -> QueryFeatures:
    """Call Gemini API with retry logic"""
    try:
        prompt = create_extraction_prompt(query)

        response = gemini_client.models.generate_content(
            model=Config.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=QueryFeatures,
            ),
        )

        if not response.parsed:
            logger.warning("Empty response from Gemini API")
            return QueryFeatures(
                objects=[],
                tags=[],
                content=query,
                people_names=[],
                location_names=[]
            )

        return response.parsed

    except Exception as e:
        logger.error("Gemini API call failed", error=str(e))
        raise


async def process_query(
    request_data: QueryProcessingRequest, request_id: str
) -> QueryProcessingResponse:
    """Process a query with proper error handling"""
    start_time = time.time()

    try:
        logger.info("Processing query", query=request_data.query, request_id=request_id)

        # Extract features with timeout
        features = await asyncio.wait_for(
            call_gemini_api(request_data.query), timeout=Config.REQUEST_TIMEOUT
        )

        # Match person names to IDs
        people_ids = mongo_client.match_person_names(
            request_data.user_id, features.people_names
        )

        # Match location names to location strings
        locations = mongo_client.match_location_names(
            request_data.user_id, features.location_names
        )

        processing_time_ms = (time.time() - start_time) * 1000

        logger.info(
            "Query processed successfully",
            processing_time_ms=processing_time_ms,
            objects_count=len(features.objects),
            tags_count=len(features.tags),
            people_count=len(people_ids),
            locations_count=len(locations),
            modalities=features.modalities,
            request_id=request_id,
        )

        return QueryProcessingResponse(
            objects=features.objects,
            tags=features.tags,
            content=features.content,
            people_ids=people_ids,
            locations=locations,
            modalities=features.modalities,
            request_id=request_id,
            processing_time_ms=processing_time_ms,
        )

    except asyncio.TimeoutError:
        logger.error("Query processing timeout", request_id=request_id)
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="Query processing timeout",
        )
    except Exception as e:
        logger.error("Query processing failed", error=str(e), request_id=request_id)
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
        if gemini_client is None or mongo_client is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not ready - clients not initialized",
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
    "/process-query",
    response_model=QueryProcessingResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        408: {"model": ErrorResponse, "description": "Request timeout"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
)
async def process_user_query(request_data: QueryProcessingRequest, http_request: Request):
    """
    Process a user search query to extract structured information.

    Extracts:
    - Objects: Entities mentioned in the query
    - Tags: Categories and descriptors
    - Content: Normalized query text
    - People IDs: Matched person IDs from MongoDB
    - Locations: Matched location strings from MongoDB (e.g., "New York, NY")
    - Modalities: Media types mentioned (video, image, text)
    """
    request_id = http_request.headers.get(
        "x-request-id", f"req_{int(time.time() * 1000)}"
    )

    try:
        if gemini_client is None or mongo_client is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service unavailable - clients not initialized",
            )

        result = await process_query(request_data, request_id)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Query processing failed", error=str(e), request_id=request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process query",
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
