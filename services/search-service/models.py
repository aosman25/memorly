from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Request model for semantic search"""
    query: str = Field(..., min_length=1, description="User search query text")
    user_id: str = Field(..., min_length=1, description="User ID for filtering")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of results to return")
    offset: int = Field(default=0, ge=0, description="Offset for pagination")


class SearchResultItem(BaseModel):
    """Individual search result item"""
    id: str = Field(..., description="Media ID")
    score: float = Field(..., description="Similarity score")
    media_type: str = Field(..., description="Media type: image, video, or text")
    source_path: Optional[str] = Field(None, description="Source path or scene ID for videos")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SearchResponse(BaseModel):
    """Response model for search results"""
    results: List[SearchResultItem] = Field(default_factory=list, description="List of search results")
    total: int = Field(..., description="Total number of results")
    query_info: Dict[str, Any] = Field(..., description="Query processing information")
    request_id: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str = "1.0.0"
    milvus_connected: bool
    query_service_connected: bool
    embed_service_connected: bool


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    request_id: str
    timestamp: str
