from pydantic import BaseModel, Field
from typing import List, Optional


class Face(BaseModel):
    """Represents a detected face with its embedding"""
    face_image: str = Field(..., description="Base64 encoded face image")
    embedding: List[float] = Field(..., description="Face embedding vector")
    confidence: float = Field(..., description="Detection confidence score")
    facial_area: dict = Field(..., description="Bounding box coordinates")


class ImageInput(BaseModel):
    """Input for a single image"""
    url: Optional[str] = Field(None, description="URL to the image")
    base64: Optional[str] = Field(None, description="Base64 encoded image")


class ExtractionRequest(BaseModel):
    """Request to extract faces from images"""
    images: List[ImageInput] = Field(..., description="List of images to process")
    similarity_threshold: Optional[float] = Field(
        0.4,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for face deduplication"
    )


class ExtractionResponse(BaseModel):
    """Response containing extracted faces"""
    faces: List[Face] = Field(..., description="List of unique detected faces")
    total_faces_detected: int = Field(..., description="Total faces before deduplication")
    unique_faces: int = Field(..., description="Number of unique faces after deduplication")
    images_processed: int = Field(..., description="Number of images successfully processed")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    deepface_loaded: bool = Field(..., description="Whether DeepFace models are loaded")


class ErrorResponse(BaseModel):
    """Error response"""
    detail: str = Field(..., description="Error message")
