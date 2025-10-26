import logging
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from pymilvus import MilvusClient
from dotenv import load_dotenv
import os
from utils import create_memory_schema, create_memory_index_params
import uvicorn

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Memory Database API",
    description="API for storing and managing user memories in Milvus",
    version="1.0.0"
)

# Initialize Milvus client
client = None

@app.on_event("startup")
async def startup_event():
    """Initialize Milvus client on startup."""
    global client
    milvus_host = os.getenv("MILVUS_HOST", "localhost")
    milvus_port = os.getenv("MILVUS_PORT", "19530")
    milvus_uri = f"{milvus_host}:{milvus_port}"
    milvus_token = os.getenv("MILVUS_TOKEN", "root:Milvus")

    logger.info(f"Connecting to Milvus at {milvus_uri}...")
    client = MilvusClient(
        uri=f"http://{milvus_uri}",
        token=milvus_token
    )
    logger.info("Successfully connected to Milvus!")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    global client
    if client:
        logger.info("Closing Milvus connection...")


# Pydantic models for request/response
class MemoryData(BaseModel):
    """Model for a single memory record."""
    id: str
    modality: str
    content: str
    embedding: List[float] = Field(..., min_length=512, max_length=512)
    timestamp: int = Field(..., description="Unix timestamp (seconds since epoch)")
    location: str
    people: List[str] = Field(default_factory=list)
    objects: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    start_timestamp_video: Optional[float] = Field(None, description="Start timestamp in seconds (video only)")
    end_timestamp_video: Optional[float] = Field(None, description="End timestamp in seconds (video only)")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "mem_20251024_174233",
                "modality": "video",
                "content": "Ali and Ahmed discussing AI ethics at a café in Istanbul.",
                "embedding": [0.012, -0.045] + [0.0] * 510,
                "timestamp": 1724174530,
                "location": "Istanbul, Turkey",
                "people": ["Ali", "Ahmed"],
                "objects": ["table", "coffee", "notebook"],
                "tags": ["conversation", "AI", "cafe", "travel"],
                "start_timestamp_video": 0.0,
                "end_timestamp_video": 5.2
            }
        }


class UpsertRequest(BaseModel):
    """Request model for upserting memories."""
    user_id: str = Field(..., description="UUID of the user")
    memories: List[MemoryData] = Field(..., description="List of memory records to upsert")


class UpsertResponse(BaseModel):
    """Response model for upsert operation."""
    status: str
    message: str
    user_id: str
    collection_name: str
    upserted_count: int
    collection_existed: bool


def ensure_user_collection(user_id: str) -> tuple[str, bool]:
    """
    Ensure a collection exists for the given user UUID.

    Args:
        user_id: UUID of the user

    Returns:
        Tuple of (collection_name, collection_existed)
    """
    # Sanitize user_id to create valid Milvus collection name
    # Milvus only allows letters, numbers, and underscores
    collection_name = user_id.replace("-", "_")
    collection_existed = False

    # Check if collection already exists
    existing_collections = client.list_collections()

    if collection_name in existing_collections:
        logger.info(f"Collection '{collection_name}' already exists for user {user_id}")
        collection_existed = True
    else:
        logger.info(f"Creating new collection '{collection_name}' for user {user_id}")

        # Create schema and index params
        schema = create_memory_schema(client)
        index_params = create_memory_index_params(client)

        # Create the collection
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )
        logger.info(f"Successfully created collection '{collection_name}'")

    return collection_name, collection_existed


@app.post("/upsert", response_model=UpsertResponse)
async def upsert_memories(request: UpsertRequest):
    """
    Upsert memory records for a specific user.

    This endpoint:
    1. Takes a user UUID
    2. Checks if a collection with that UUID exists
    3. Creates the collection if it doesn't exist
    4. Upserts the memory data into the collection
    """
    try:
        logger.info(f"Processing upsert request for user: {request.user_id}")
        logger.info(f"Number of memories to upsert: {len(request.memories)}")

        # Ensure collection exists for this user
        collection_name, collection_existed = ensure_user_collection(request.user_id)

        # Convert Pydantic models to dictionaries for Milvus
        memory_dicts = [memory.model_dump() for memory in request.memories]

        # Handle None values for optional float fields
        # Milvus doesn't accept None for FLOAT fields, convert to 0.0
        for memory in memory_dicts:
            if memory.get('start_timestamp_video') is None:
                memory['start_timestamp_video'] = 0.0
            if memory.get('end_timestamp_video') is None:
                memory['end_timestamp_video'] = 0.0

        # Upsert data into the collection
        client.upsert(
            collection_name=collection_name,
            data=memory_dicts
        )

        logger.info(f"Successfully upserted {len(memory_dicts)} memories for user {request.user_id}")

        return UpsertResponse(
            status="success",
            message=f"Successfully upserted {len(memory_dicts)} memories",
            user_id=request.user_id,
            collection_name=collection_name,
            upserted_count=len(memory_dicts),
            collection_existed=collection_existed
        )

    except Exception as e:
        logger.error(f"Error upserting memories for user {request.user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upsert memories: {str(e)}"
        )


@app.get("/collections")
async def list_collections():
    """List all collections in the database."""
    try:
        collections = client.list_collections()
        return {
            "status": "success",
            "collections": collections,
            "count": len(collections)
        }
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list collections: {str(e)}"
        )


@app.get("/collection/{user_id}")
async def get_collection_info(user_id: str):
    """Get information about a specific user's collection."""
    try:
        collection_name = user_id.replace("-", "_")
        collections = client.list_collections()

        if collection_name not in collections:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection for user {user_id} does not exist"
            )

        # Get collection stats
        stats = client.get_collection_stats(collection_name=collection_name)

        return {
            "status": "success",
            "user_id": user_id,
            "collection_name": collection_name,
            "stats": stats
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting collection info for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get collection info: {str(e)}"
        )


@app.delete("/collection/{user_id}")
async def delete_user_collection(user_id: str):
    """Delete a user's collection."""
    try:
        collection_name = user_id.replace("-", "_")
        collections = client.list_collections()

        if collection_name not in collections:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection for user {user_id} does not exist"
            )

        client.drop_collection(collection_name=collection_name)
        logger.info(f"Deleted collection '{collection_name}' for user {user_id}")

        return {
            "status": "success",
            "message": f"Successfully deleted collection for user {user_id}",
            "user_id": user_id,
            "collection_name": collection_name
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting collection for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete collection: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Memory Database API",
        "milvus_connected": client is not None
    }


if __name__ == "__main__":
    # Run the server
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting Memory Database API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
