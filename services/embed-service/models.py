from typing import List, Optional
from pydantic import BaseModel, Field


class VideoFrame(BaseModel):
    """Model for a video frame with optional transcript"""
    frame_base64: str = Field(..., description="Base64 encoded image")
    transcript: Optional[str] = Field(None, description="Transcript for this frame")


class ImageEmbedRequest(BaseModel):
    """Request model for embedding a single image"""
    image_base64: str = Field(..., description="Base64 encoded image")


class VideoEmbedRequest(BaseModel):
    """Request model for embedding video frames"""
    frames: List[VideoFrame] = Field(..., min_items=1, description="List of video frames with transcripts")
    visual_weight: float = Field(0.6, description="Weight for visual embeddings", ge=0.0, le=1.0)
    text_weight: float = Field(0.4, description="Weight for text embeddings", ge=0.0, le=1.0)


class TextEmbedRequest(BaseModel):
    """Request model for embedding plain text"""
    text: str = Field(..., min_length=1, description="Text to embed")


class BatchImageEmbedRequest(BaseModel):
    """Request model for batch image embedding"""
    images: List[str] = Field(..., min_items=1, max_items=100, description="List of base64 encoded images")


class EmbedResponse(BaseModel):
    """Response model for single embedding"""
    embedding: List[float] = Field(..., description="512-dimensional embedding vector")
    request_id: str = Field(..., description="Request identifier")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class BatchEmbedResponse(BaseModel):
    """Response model for batch embedding"""
    embeddings: List[List[float]] = Field(..., description="List of 512-dimensional embedding vectors")
    count: int = Field(..., description="Number of embeddings")
    request_id: str = Field(..., description="Request identifier")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str = "1.0.0"
    model: str


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    request_id: str
    timestamp: str
