from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"


class ProcessMediaRequest(BaseModel):
    media_id: str = Field(..., description="Unique identifier for the media")
    user_id: str = Field(..., description="User UUID")
    media_type: MediaType = Field(..., description="Type of media")
    timestamp: int = Field(..., description="Unix timestamp when media was created")
    location: Optional[str] = Field(None, description="Location where media was captured")


class ProcessMediaResponse(BaseModel):
    success: bool
    media_id: str
    message: str
    persons_created: Optional[int] = None
    persons_updated: Optional[int] = None
    embedding_dimension: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    services: dict


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query text")
    user_id: str = Field(..., description="User UUID")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
    offset: int = Field(default=0, ge=0, description="Offset for pagination")


class SearchResultItem(BaseModel):
    id: str
    score: float
    media_type: str
    source_path: Optional[str] = None
    metadata: dict


class SearchResponse(BaseModel):
    success: bool
    results: List[SearchResultItem]
    total: int
    query_info: dict
    processing_time_ms: float
    message: Optional[str] = None
