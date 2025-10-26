import logging
import uuid
import httpx
from typing import Dict, List, Tuple, Optional
from service_clients import ServiceClients
from mongodb_client import MongoDBClient
from config import Config

logger = logging.getLogger(__name__)


class MediaPipeline:
    """Orchestrates the processing pipeline for different media types."""

    def __init__(self, service_clients: ServiceClients, mongo_client: MongoDBClient):
        self.service_clients = service_clients
        self.mongo_client = mongo_client

    async def process_faces(
        self,
        user_id: str,
        media_id: str,
        media_path: str,
        is_video: bool = False
    ) -> Tuple[int, int, List[str]]:
        """
        Process faces from media and update persons in MongoDB.

        Args:
            user_id: User UUID
            media_id: Media ID
            media_path: Path to media file
            is_video: Whether media is a video

        Returns:
            Tuple of (persons_created, persons_updated, person_ids)
        """
        persons_created = 0
        persons_updated = 0
        person_ids = []

        try:
            # Extract faces
            faces = await self.service_clients.extract_faces(media_path, is_video)

            if not faces:
                logger.info(f"No faces found in media: {media_id}")
                return persons_created, persons_updated, person_ids

            # Process each face
            for face in faces:
                face_embedding = face.get("embedding")
                if not face_embedding:
                    logger.warning("Face without embedding, skipping")
                    continue

                # Check if this face matches an existing person
                matching_person_id = self.mongo_client.find_matching_person(
                    user_id=user_id,
                    face_embedding=face_embedding,
                    threshold=Config.FACE_SIMILARITY_THRESHOLD
                )

                if matching_person_id:
                    # Update existing person
                    self.mongo_client.add_media_to_person(
                        user_id=user_id,
                        person_id=matching_person_id,
                        media_id=media_id
                    )
                    persons_updated += 1
                    person_ids.append(matching_person_id)
                else:
                    # Create new person
                    new_person_id = str(uuid.uuid4())
                    self.mongo_client.create_person(
                        user_id=user_id,
                        person_id=new_person_id,
                        face_embedding=face_embedding,
                        media_id=media_id
                    )
                    persons_created += 1
                    person_ids.append(new_person_id)

            logger.info(f"Face processing complete: {persons_created} created, {persons_updated} updated")

        except Exception as e:
            logger.error(f"Error processing faces: {e}", exc_info=True)
            raise

        return persons_created, persons_updated, person_ids

    async def process_image(
        self,
        user_id: str,
        media_id: str,
        media_path: str,
        timestamp: int,
        location: Optional[str] = None
    ) -> Dict:
        """
        Process image through the full pipeline.

        Pipeline:
        1. Extract features (objects, content, tags)
        2. Extract and process faces
        3. Generate embedding
        4. Upsert to vector database

        Args:
            user_id: User UUID
            media_id: Media ID
            media_path: Path to image file
            timestamp: Unix timestamp
            location: Location string

        Returns:
            Processing result dictionary
        """
        logger.info(f"Starting image pipeline for media: {media_id}")

        try:
            # Step 1: Extract features
            features = await self.service_clients.extract_features(media_path)
            objects = features.get("objects", [])
            content = features.get("content", "")
            tags = features.get("tags", [])

            # Step 2: Process faces
            persons_created, persons_updated, person_ids = await self.process_faces(
                user_id=user_id,
                media_id=media_id,
                media_path=media_path,
                is_video=False
            )

            # Step 3: Generate embedding
            embedding = await self.service_clients.embed_image(media_path)

            # Step 4: Upsert to vector database
            await self.service_clients.upsert_to_vector_db(
                user_id=user_id,
                media_id=media_id,
                embedding=embedding,
                timestamp=timestamp,
                modality="image",
                source_path=media_id,  # For images, source_path equals media_id
                objects=objects,
                content=content,
                tags=tags,
                location=location,
                people=person_ids
            )

            logger.info(f"Image pipeline completed successfully for media: {media_id}")

            return {
                "success": True,
                "media_id": media_id,
                "message": "Image processed successfully",
                "persons_created": persons_created,
                "persons_updated": persons_updated,
                "embedding_dimension": len(embedding)
            }

        except Exception as e:
            logger.error(f"Error in image pipeline: {e}", exc_info=True)
            return {
                "success": False,
                "media_id": media_id,
                "message": f"Error processing image: {str(e)}",
                "persons_created": 0,
                "persons_updated": 0,
                "embedding_dimension": None
            }

    async def process_video(
        self,
        user_id: str,
        media_id: str,
        media_path: str,
        timestamp: int,
        location: Optional[str] = None
    ) -> Dict:
        """
        Process video through the full pipeline.

        Pipeline:
        1. Segment video into scenes
        2. Extract features from representative frames
        3. Extract and process faces
        4. Generate fused embedding (visual + text)
        5. Upsert to vector database

        Args:
            user_id: User UUID
            media_id: Media ID
            media_path: Path to video file
            timestamp: Unix timestamp
            location: Location string

        Returns:
            Processing result dictionary
        """
        logger.info(f"Starting video pipeline for media: {media_id}")

        try:
            # Step 1: Segment video
            scenes = await self.service_clients.segment_video(media_path)

            if not scenes:
                logger.warning(f"No scenes detected in video: {media_id}")
                return {
                    "success": False,
                    "media_id": media_id,
                    "message": "No scenes detected in video",
                    "persons_created": 0,
                    "persons_updated": 0,
                    "embedding_dimension": None
                }

            # Step 2: Extract faces from scene frames
            # Collect all scene frames for face detection
            scene_frames = []
            for scene in scenes:
                frame_base64 = scene.get("frame_base64")
                if frame_base64:
                    scene_frames.append({"base64": frame_base64})

            # Extract faces from all frames
            all_faces = []
            if scene_frames:
                async with httpx.AsyncClient(timeout=self.service_clients.timeout) as client:
                    response = await client.post(
                        f"{Config.FACE_EXTRACTION_SERVICE_URL}/extract-faces",
                        json={"images": scene_frames}
                    )
                    response.raise_for_status()
                    result = response.json()
                    all_faces = result.get("faces", [])

            # Process faces and match to persons in database
            persons_created = 0
            persons_updated = 0
            person_ids = []

            for face in all_faces:
                face_embedding = face.get("embedding")
                if not face_embedding:
                    continue

                # Try to match face to existing person (returns person_id string or None)
                matched_person_id = self.mongo_client.find_matching_person(
                    user_id=user_id,
                    face_embedding=face_embedding
                )

                if matched_person_id:
                    # Update existing person
                    self.mongo_client.add_media_to_person(
                        user_id=user_id,
                        person_id=matched_person_id,
                        media_id=media_id
                    )
                    person_ids.append(matched_person_id)
                    persons_updated += 1
                    logger.info(f"Matched face to existing person: {matched_person_id}")
                else:
                    # Create new person
                    new_person_id = str(uuid.uuid4())
                    self.mongo_client.create_person(
                        user_id=user_id,
                        person_id=new_person_id,
                        face_embedding=face_embedding,
                        media_id=media_id
                    )
                    person_ids.append(new_person_id)
                    persons_created += 1
                    logger.info(f"Created new person: {new_person_id}")

            logger.info(f"Face processing complete: {persons_created} created, {persons_updated} updated")

            # Step 3: Process each scene separately
            scenes_processed = 0
            for idx, scene in enumerate(scenes, 1):
                # Generate unique ID for this scene
                scene_id = f"{media_id}_scene_{idx}"

                # Get scene timestamps
                start_timestamp_video = scene.get("start_timestamp", 0.0)
                end_timestamp_video = scene.get("end_timestamp", 0.0)

                # Get transcript
                transcript = scene.get("transcript", "")

                # Extract features from scene frame
                frame_base64 = scene.get("frame_base64")
                if frame_base64:
                    features = await self.service_clients.extract_features_from_base64(frame_base64)
                    objects = features.get("objects", [])
                    tags = features.get("tags", [])
                    description = features.get("content", "")
                else:
                    logger.warning(f"No frame available in scene {idx}, skipping feature extraction")
                    objects = []
                    tags = []
                    description = ""

                # Build content with description and transcript
                content_parts = []
                if description:
                    content_parts.append(f"[Description]\n{description}")
                if transcript:
                    content_parts.append(f"[Transcript]\n{transcript}")
                content = "\n\n".join(content_parts) if content_parts else ""

                # Generate embedding for this scene
                embedding = await self.service_clients.embed_video([scene])

                # Upsert this scene to vector database
                await self.service_clients.upsert_to_vector_db(
                    user_id=user_id,
                    media_id=scene_id,  # Unique ID for this scene
                    embedding=embedding,
                    timestamp=timestamp,
                    modality="video",
                    source_path=media_id,  # Original video ID
                    objects=objects,
                    content=content,
                    tags=tags,
                    location=location,
                    people=person_ids,
                    start_timestamp_video=start_timestamp_video,
                    end_timestamp_video=end_timestamp_video
                )

                scenes_processed += 1
                logger.info(f"Processed scene {idx}/{len(scenes)} for video {media_id}")

            logger.info(f"Video pipeline completed successfully for media: {media_id}")

            return {
                "success": True,
                "media_id": media_id,
                "message": f"Video processed successfully ({scenes_processed} scenes)",
                "persons_created": persons_created,
                "persons_updated": persons_updated,
                "embedding_dimension": 512  # All embeddings are 512-dimensional
            }

        except Exception as e:
            logger.error(f"Error in video pipeline: {e}", exc_info=True)
            return {
                "success": False,
                "media_id": media_id,
                "message": f"Error processing video: {str(e)}",
                "persons_created": 0,
                "persons_updated": 0,
                "embedding_dimension": None
            }

    async def process_text(
        self,
        user_id: str,
        media_id: str,
        text_content: str,
        timestamp: int,
        location: Optional[str] = None
    ) -> Dict:
        """
        Process text through the pipeline.

        Pipeline:
        1. Generate text embedding
        2. Upsert to vector database

        Args:
            user_id: User UUID
            media_id: Media ID
            text_content: Text content to process
            timestamp: Unix timestamp
            location: Location string

        Returns:
            Processing result dictionary
        """
        logger.info(f"Starting text pipeline for media: {media_id}")

        try:
            # Step 1: Generate embedding
            embedding = await self.service_clients.embed_text(text_content)

            # Step 2: Upsert to vector database
            # For text, we use the text itself as content
            await self.service_clients.upsert_to_vector_db(
                user_id=user_id,
                media_id=media_id,
                embedding=embedding,
                timestamp=timestamp,
                modality="text",
                source_path=media_id,  # For text, source_path equals media_id
                objects=[],
                content=text_content,
                tags=[],
                location=location,
                people=[]
            )

            logger.info(f"Text pipeline completed successfully for media: {media_id}")

            return {
                "success": True,
                "media_id": media_id,
                "message": "Text processed successfully",
                "persons_created": None,
                "persons_updated": None,
                "embedding_dimension": len(embedding)
            }

        except Exception as e:
            logger.error(f"Error in text pipeline: {e}", exc_info=True)
            return {
                "success": False,
                "media_id": media_id,
                "message": f"Error processing text: {str(e)}",
                "persons_created": None,
                "persons_updated": None,
                "embedding_dimension": None
            }
