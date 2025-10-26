import httpx
import base64
import logging
from typing import List, Dict, Optional, Tuple
from config import Config

logger = logging.getLogger(__name__)


class ServiceClients:
    """Client for communicating with all microservices."""

    def __init__(self):
        self.timeout = httpx.Timeout(300.0, connect=60.0)  # 5 min timeout for processing

    async def extract_features(self, image_path: str) -> Dict:
        """
        Call extract-features-service to extract objects, content, and tags from an image.

        Args:
            image_path: Path to the image file

        Returns:
            Dict with keys: objects, content, tags
        """
        logger.info(f"Extracting features from image: {image_path}")

        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{Config.EXTRACT_FEATURES_SERVICE_URL}/extract",
                json={"image_base64": image_base64}
            )
            response.raise_for_status()
            result = response.json()

        # Extract features from the nested response structure
        features = result.get('features', {})
        logger.info(f"Features extracted: {len(features.get('objects', []))} objects, {len(features.get('tags', []))} tags")
        return features

    async def extract_faces(self, media_path: str, is_video: bool = False) -> List[Dict]:
        """
        Call face-extraction-service to extract faces from image or video frames.

        Args:
            media_path: Path to the media file
            is_video: Whether the media is a video (will extract frames first)

        Returns:
            List of face objects with embeddings
        """
        logger.info(f"Extracting faces from {'video' if is_video else 'image'}: {media_path}")

        # For images, read and encode
        if not is_video:
            with open(media_path, "rb") as f:
                image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode("utf-8")

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{Config.FACE_EXTRACTION_SERVICE_URL}/extract-faces",
                    json={"images": [{"base64": image_base64}]}
                )
                response.raise_for_status()
                result = response.json()

            faces = result.get("faces", [])
        else:
            # For videos, pass the path directly (service will handle frame extraction)
            # This assumes the face extraction service accepts video paths
            with open(media_path, "rb") as f:
                video_data = f.read()
            video_base64 = base64.b64encode(video_data).decode("utf-8")

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{Config.FACE_EXTRACTION_SERVICE_URL}/extract-faces",
                    json={"images": [{"base64": video_base64}]}
                )
                response.raise_for_status()
                result = response.json()

            faces = result.get("faces", [])

        logger.info(f"Extracted {len(faces)} unique faces")
        return faces

    async def segment_video(self, video_path: str) -> List[Dict]:
        """
        Call video-segmentation-service to segment video into scenes.

        Args:
            video_path: Path to the video file

        Returns:
            List of scenes with frames and transcripts
        """
        logger.info(f"Segmenting video: {video_path}")

        with open(video_path, "rb") as f:
            video_data = f.read()
        video_base64 = base64.b64encode(video_data).decode("utf-8")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{Config.VIDEO_SEGMENTATION_SERVICE_URL}/segment",
                json={"video": video_base64}
            )
            response.raise_for_status()
            result = response.json()

        scenes = result.get("scenes", [])
        logger.info(f"Video segmented into {len(scenes)} scenes")
        return scenes

    async def embed_image(self, image_path: str) -> List[float]:
        """
        Call embed-service to generate embedding for an image.

        Args:
            image_path: Path to the image file

        Returns:
            Image embedding vector
        """
        logger.info(f"Generating embedding for image: {image_path}")

        with open(image_path, "rb") as f:
            image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{Config.EMBED_SERVICE_URL}/embed/image",
                json={"image_base64": image_base64}
            )
            response.raise_for_status()
            result = response.json()

        embedding = result.get("embedding", [])
        logger.info(f"Generated image embedding with dimension: {len(embedding)}")
        return embedding

    async def embed_text(self, text: str) -> List[float]:
        """
        Call embed-service to generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Text embedding vector
        """
        logger.info(f"Generating embedding for text ({len(text)} chars)")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{Config.EMBED_SERVICE_URL}/embed/text",
                json={"text": text}
            )
            response.raise_for_status()
            result = response.json()

        embedding = result.get("embedding", [])
        logger.info(f"Generated text embedding with dimension: {len(embedding)}")
        return embedding

    async def embed_video(self, scenes: List[Dict]) -> List[float]:
        """
        Call embed-service to generate fused embedding for video scenes.

        Args:
            scenes: List of scenes with frames and transcripts

        Returns:
            Fused video embedding vector
        """
        logger.info(f"Generating fused embedding for {len(scenes)} video scenes")

        # Prepare frames with transcripts
        frames = []
        for scene in scenes:
            frames.append({
                "frame_base64": scene.get("frame"),  # base64 encoded frame
                "transcript": scene.get("transcript", "")
            })

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{Config.EMBED_SERVICE_URL}/embed/video",
                json={
                    "frames": frames,
                    "visual_weight": Config.VIDEO_VISUAL_WEIGHT,
                    "text_weight": Config.VIDEO_TEXT_WEIGHT
                }
            )
            response.raise_for_status()
            result = response.json()

        embedding = result.get("embedding", [])
        logger.info(f"Generated video embedding with dimension: {len(embedding)}")
        return embedding

    async def upsert_to_vector_db(
        self,
        user_id: str,
        media_id: str,
        embedding: List[float],
        timestamp: int,
        modality: str = "image",
        source_path: Optional[str] = None,
        objects: Optional[List[str]] = None,
        content: Optional[str] = None,
        tags: Optional[List[str]] = None,
        location: Optional[str] = None,
        people: Optional[List[str]] = None,
        start_timestamp_video: Optional[float] = None,
        end_timestamp_video: Optional[float] = None
    ) -> Dict:
        """
        Call upsert-service to insert/update embedding in Milvus vector database.

        Args:
            user_id: User UUID
            media_id: Media ID
            embedding: Embedding vector
            timestamp: Unix timestamp
            modality: Type of media (image/video/text)
            source_path: Source media ID (same as media_id for images, original video_id for segments)
            objects: List of objects in the media
            content: Descriptive content
            tags: List of tags
            location: Location string
            people: List of people in the media
            start_timestamp_video: Start timestamp for video (optional)
            end_timestamp_video: End timestamp for video (optional)

        Returns:
            Response from upsert service
        """
        logger.info(f"Upserting to vector DB: user={user_id}, media={media_id}")

        memory_data = {
            "id": media_id,
            "modality": modality,
            "content": content or "",
            "embedding": embedding,
            "timestamp": timestamp,
            "location": location or "",
            "people": people or [],
            "objects": objects or [],
            "tags": tags or [],
            "source_path": source_path or media_id  # Default to media_id if not provided
        }

        # Add video timestamps if provided
        if start_timestamp_video is not None:
            memory_data["start_timestamp_video"] = start_timestamp_video
        if end_timestamp_video is not None:
            memory_data["end_timestamp_video"] = end_timestamp_video

        payload = {
            "user_id": user_id,
            "memories": [memory_data]  # Changed to plural and wrapped in list
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{Config.UPSERT_SERVICE_URL}/upsert",
                json=payload
            )
            response.raise_for_status()
            result = response.json()

        logger.info(f"Successfully upserted to vector DB")
        return result

    async def health_check(self, service_url: str) -> bool:
        """Check if a service is healthy."""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                response = await client.get(f"{service_url}/health")
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"Health check failed for {service_url}: {e}")
            return False
