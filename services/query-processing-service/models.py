from typing import List, Optional
from pydantic import BaseModel, Field


class QueryFeatures(BaseModel):
    """Extracted features from a user query"""
    objects: List[str] = Field(default_factory=list, description="List of objects mentioned in the query")
    tags: List[str] = Field(default_factory=list, description="Relevant tags/categories from the query")
    content: str = Field(..., min_length=1, description="Cleaned and normalized query content")
    people_names: List[str] = Field(default_factory=list, description="Names of people mentioned in the query")
    location_names: List[str] = Field(default_factory=list, description="Location names mentioned in the query")
    modalities: List[str] = Field(default_factory=list, description="Media type modalities: video, image, text")


class QueryProcessingRequest(BaseModel):
    """Request model for query processing"""
    query: str = Field(..., min_length=1, description="User search query text")
    user_id: str = Field(..., min_length=1, description="User ID for person/location lookup")


class QueryProcessingResponse(BaseModel):
    """Response model for query processing with matched IDs"""
    objects: List[str] = Field(default_factory=list, description="List of objects")
    tags: List[str] = Field(default_factory=list, description="Relevant tags/categories")
    content: str = Field(..., description="Cleaned query content")
    people_ids: List[str] = Field(default_factory=list, description="Matched person IDs from MongoDB")
    locations: List[str] = Field(default_factory=list, description="Matched location strings from MongoDB")
    modalities: List[str] = Field(default_factory=list, description="Media type modalities: video, image, text")
    request_id: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str = "1.0.0"
    gemini_model: str


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    request_id: str
    timestamp: str
