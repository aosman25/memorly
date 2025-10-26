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
