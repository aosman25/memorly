from typing import List, Optional
from pydantic import BaseModel, Field


class VideoScene(BaseModel):
    """Model for a single video scene"""
    frame_base64: str = Field(..., description="Sample frame from the scene encoded as base64")
    transcript: str = Field(..., description="Transcript of audio during this scene")
    start_timestamp: float = Field(..., description="Start time of the scene in seconds")
    end_timestamp: float = Field(..., description="End time of the scene in seconds")
    scene_number: int = Field(..., description="Sequential scene number")


class SegmentationRequest(BaseModel):
    """Request model for video segmentation"""
    video_url: Optional[str] = Field(None, description="URL of the video to process")
    video_base64: Optional[str] = Field(None, description="Base64 encoded video data")
    scene_threshold: float = Field(0.3, description="Scene detection threshold (0-1). Lower = more sensitive", ge=0.0, le=1.0)
    language: Optional[str] = Field(None, description="Language code (ISO-639-1) for audio transcription")

    def model_post_init(self, __context):
        """Ensure at least one video source is provided"""
        if not self.video_url and not self.video_base64:
            raise ValueError("Either video_url or video_base64 must be provided")


class SegmentationResponse(BaseModel):
    """Response model for video segmentation"""
    scenes: List[VideoScene] = Field(..., description="List of detected scenes")
    total_scenes: int = Field(..., description="Total number of scenes detected")
    video_duration: float = Field(..., description="Total video duration in seconds")
    request_id: str = Field(..., description="Request identifier")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str = "1.0.0"
    ffmpeg_available: bool
    whisper_model: str


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    request_id: str
    timestamp: str
