import os
from typing import Optional


class Config:
    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # MongoDB Configuration
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "memorly")

    # Service URLs
    EXTRACT_FEATURES_SERVICE_URL: str = os.getenv(
        "EXTRACT_FEATURES_SERVICE_URL",
        "http://localhost:8001"
    )
    FACE_EXTRACTION_SERVICE_URL: str = os.getenv(
        "FACE_EXTRACTION_SERVICE_URL",
        "http://localhost:8002"
    )
    EMBED_SERVICE_URL: str = os.getenv(
        "EMBED_SERVICE_URL",
        "http://localhost:8003"
    )
    UPSERT_SERVICE_URL: str = os.getenv(
        "UPSERT_SERVICE_URL",
        "http://localhost:8004"
    )
    VIDEO_SEGMENTATION_SERVICE_URL: str = os.getenv(
        "VIDEO_SEGMENTATION_SERVICE_URL",
        "http://localhost:8005"
    )
    QUERY_PROCESSING_SERVICE_URL: str = os.getenv(
        "QUERY_PROCESSING_SERVICE_URL",
        "http://localhost:8006"
    )
    SEARCH_SERVICE_URL: str = os.getenv(
        "SEARCH_SERVICE_URL",
        "http://localhost:8007"
    )

    # Face Similarity Threshold
    FACE_SIMILARITY_THRESHOLD: float = float(
        os.getenv("FACE_SIMILARITY_THRESHOLD", "0.4")
    )

    # Video Processing Parameters
    VIDEO_VISUAL_WEIGHT: float = float(os.getenv("VIDEO_VISUAL_WEIGHT", "0.6"))
    VIDEO_TEXT_WEIGHT: float = float(os.getenv("VIDEO_TEXT_WEIGHT", "0.4"))

    # File Upload
    MAX_UPLOAD_SIZE: int = int(os.getenv("MAX_UPLOAD_SIZE", "100000000"))  # 100MB
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "/tmp/gateway-uploads")

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
