import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import List, Dict, Any

import httpx
import structlog
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymilvus import MilvusClient, Collection
from dotenv import load_dotenv

from models import (
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    HealthResponse,
    ErrorResponse,
)

# Load environment variables
load_dotenv()


# Configuration
class Config:
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
    MILVUS_URI = f"{MILVUS_HOST}:{MILVUS_PORT}"
    MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "root:Milvus")

    QUERY_PROCESSING_SERVICE_URL = os.getenv("QUERY_PROCESSING_SERVICE_URL", "http://localhost:8006")
    EMBED_SERVICE_URL = os.getenv("EMBED_SERVICE_URL", "http://localhost:8003")

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))
    PORT = int(os.getenv("PORT", "8000"))

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        required = ["MILVUS_HOST", "MILVUS_PORT"]
        missing = [var for var in required if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


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
milvus_client = None
http_client = None
logger = structlog.get_logger()


# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global milvus_client, http_client

    # Startup
    try:
        logger.info("Starting Search Service")
        Config.validate()

        # Initialize Milvus client
        milvus_client = MilvusClient(
            uri=f"http://{Config.MILVUS_URI}",
            token=Config.MILVUS_TOKEN
        )
        logger.info("Connected to Milvus", uri=Config.MILVUS_URI)

        # Initialize HTTP client
        http_client = httpx.AsyncClient(timeout=httpx.Timeout(Config.REQUEST_TIMEOUT))

        logger.info("Service started successfully")
        yield

    except Exception as e:
        logger.error("Failed to start service", error=str(e))
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Search Service")
        if http_client:
            await http_client.aclose()
        milvus_client = None


# FastAPI app
app = FastAPI(
    title="Search API",
    description="Semantic search with intelligent filtering using vector database",
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


async def process_query(query: str, user_id: str) -> Dict[str, Any]:
    """Process query using query-processing-service"""
    try:
        response = await http_client.post(
            f"{Config.QUERY_PROCESSING_SERVICE_URL}/process-query",
            json={"query": query, "user_id": user_id}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error("Query processing failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Query processing service error: {str(e)}"
        )


async def get_embedding(text: str) -> List[float]:
    """Get embedding from embed-service"""
    try:
        response = await http_client.post(
            f"{Config.EMBED_SERVICE_URL}/embed/text",
            json={"text": text}
        )
        response.raise_for_status()
        result = response.json()
        return result["embedding"]
    except Exception as e:
        logger.error("Embedding generation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embed service error: {str(e)}"
        )


def build_filter_expression(query_info: Dict[str, Any]) -> str:
    """Build Milvus filter expression from query information (people and locations only)"""
    filters = []

    # Filter by people IDs
    people_ids = query_info.get("people_ids", [])
    if people_ids:
        # Check if people array contains any of the mentioned people
        people_conditions = [f'ARRAY_CONTAINS(people, "{pid}")' for pid in people_ids]
        filters.append(f'({" OR ".join(people_conditions)})')

    # Filter by locations
    locations = query_info.get("locations", [])
    if locations:
        # Match location field with any of the mentioned locations
        location_conditions = [f'location == "{loc}"' for loc in locations]
        filters.append(f'({" OR ".join(location_conditions)})')

    # Combine all filters with AND
    if len(filters) == 0:
        return ""  # No filters
    elif len(filters) == 1:
        return filters[0]
    else:
        return " AND ".join(filters)


async def perform_search(
    user_id: str,
    embedding: List[float],
    filter_expr: str,
    limit: int,
    offset: int
) -> List[Dict[str, Any]]:
    """Perform vector similarity search in Milvus"""
    try:
        search_params = {
            "metric_type": "IP",  # Inner Product
            "params": {"nprobe": 10}
        }

        # Use user_id as collection name
        collection_name = user_id

        # Perform search (using COSINE metric to match collection configuration)
        results = milvus_client.search(
            collection_name=collection_name,
            data=[embedding],
            anns_field="embedding",
            search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=limit + offset,  # Get more to handle offset
            output_fields=["id", "modality", "content", "source_path", "timestamp", "location", "people", "objects", "tags", "start_timestamp_video", "end_timestamp_video"],
            filter=filter_expr if filter_expr else None
        )

        # Extract and format results
        formatted_results = []
        if results and len(results) > 0:
            # Skip offset and take limit
            for hit in results[0][offset:offset + limit]:
                # Milvus returns data in hit["entity"] structure
                entity = hit.get("entity", {})

                # Parse JSON string fields or convert to lists (people, objects, tags can be JSON strings or protobuf arrays)
                import json as json_lib
                people = entity.get("people", [])
                objects = entity.get("objects", [])
                tags = entity.get("tags", [])

                # Convert to list - handles both JSON strings and protobuf RepeatedScalarContainer
                if isinstance(people, str):
                    people_list = json_lib.loads(people) if people else []
                else:
                    people_list = list(people) if people else []

                if isinstance(objects, str):
                    objects_list = json_lib.loads(objects) if objects else []
                else:
                    objects_list = list(objects) if objects else []

                if isinstance(tags, str):
                    tags_list = json_lib.loads(tags) if tags else []
                else:
                    tags_list = list(tags) if tags else []

                formatted_results.append({
                    "id": hit.get("id"),
                    "score": float(hit.get("distance", 0)),
                    "media_type": entity.get("modality", "unknown"),
                    "source_path": entity.get("source_path"),
                    "metadata": {
                        "content": entity.get("content"),
                        "timestamp": entity.get("timestamp"),
                        "location": entity.get("location"),
                        "person_ids": people_list,
                        "objects": objects_list,
                        "tags": tags_list,
                        "start_timestamp_video": entity.get("start_timestamp_video"),
                        "end_timestamp_video": entity.get("end_timestamp_video"),
                    }
                })

        return formatted_results

    except Exception as e:
        logger.error("Milvus search failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    milvus_ok = milvus_client is not None

    # Check query service
    query_service_ok = False
    try:
        response = await http_client.get(f"{Config.QUERY_PROCESSING_SERVICE_URL}/health", timeout=5)
        query_service_ok = response.status_code == 200
    except:
        pass

    # Check embed service
    embed_service_ok = False
    try:
        response = await http_client.get(f"{Config.EMBED_SERVICE_URL}/health", timeout=5)
        embed_service_ok = response.status_code == 200
    except:
        pass

    return HealthResponse(
        status="healthy" if milvus_ok else "degraded",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        milvus_connected=milvus_ok,
        query_service_connected=query_service_ok,
        embed_service_connected=embed_service_ok,
    )


@app.post(
    "/search",
    response_model=SearchResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
)
async def search(request_data: SearchRequest, http_request: Request):
    """
    Perform semantic search with intelligent filtering.

    Steps:
    1. Process query using query-processing-service to extract structured information
    2. Generate embedding from the normalized content using embed-service
    3. Build filter expression from extracted metadata (people, locations, modalities)
    4. Execute vector similarity search in Milvus with filters
    5. Return ranked results
    """
    request_id = http_request.headers.get(
        "x-request-id", f"req_{int(time.time() * 1000)}"
    )
    start_time = time.time()

    try:
        if milvus_client is None or http_client is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not ready - clients not initialized",
            )

        logger.info("Processing search request", query=request_data.query, request_id=request_id)

        # Step 1: Process query
        query_info = await process_query(request_data.query, request_data.user_id)
        logger.info("Query processed", query_info=query_info, request_id=request_id)

        # Step 2: Generate embedding from content
        content = query_info.get("content", request_data.query)
        embedding = await get_embedding(content)
        logger.info("Embedding generated", dimension=len(embedding), request_id=request_id)

        # Step 3: Build filter expression
        filter_expr = build_filter_expression(query_info)
        logger.info("Filter built", filter=filter_expr, request_id=request_id)

        # Step 4: Perform search (using user_id as collection name)
        results = await perform_search(
            user_id=request_data.user_id,
            embedding=embedding,
            filter_expr=filter_expr,
            limit=request_data.limit,
            offset=request_data.offset
        )

        processing_time_ms = (time.time() - start_time) * 1000

        logger.info(
            "Search completed",
            results_count=len(results),
            processing_time_ms=processing_time_ms,
            request_id=request_id,
        )

        # Convert to response model
        result_items = [SearchResultItem(**r) for r in results]

        return SearchResponse(
            results=result_items,
            total=len(result_items),
            query_info={
                "objects": query_info.get("objects", []),
                "tags": query_info.get("tags", []),
                "content": content,
                "people_ids": query_info.get("people_ids", []),
                "locations": query_info.get("locations", []),
                "modalities": query_info.get("modalities", []),
            },
            request_id=request_id,
            processing_time_ms=processing_time_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Search failed", error=str(e), request_id=request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform search",
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
