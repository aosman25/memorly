from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class ImageFeatures(BaseModel):
    """Extracted features from an image"""
    objects: List[str] = Field(default_factory=list, description="List of detected objects in the image")
    content: str = Field(..., min_length=1, description="Descriptive caption of the image")
    tags: List[str] = Field(default_factory=list, description="Relevant tags/categories for the image")


class ExtractionRequest(BaseModel):
    """Request model for feature extraction"""
    image_url: Optional[str] = Field(None, description="URL of the image to process")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image data")

    @field_validator("image_url", "image_base64")
    @classmethod
    def validate_image_source(cls, v, info):
        # At least one must be provided
        return v

    def model_post_init(self, __context):
        """Ensure at least one image source is provided"""
        if not self.image_url and not self.image_base64:
            raise ValueError("Either image_url or image_base64 must be provided")


class BatchExtractionRequest(BaseModel):
    """Request model for batch feature extraction"""
    images: List[ExtractionRequest] = Field(..., min_items=1, max_items=10, description="List of images to process")


class ExtractionResponse(BaseModel):
    """Response model for a single extraction"""
    features: ImageFeatures
    request_id: str
    processing_time_ms: float


class BatchExtractionResponse(BaseModel):
    """Response model for batch extraction"""
    results: List[ExtractionResponse]
    processed_count: int
    request_id: str
    total_processing_time_ms: float


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
