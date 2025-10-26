import logging
import uuid
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

            # Step 2: Extract features and content from scenes
            # Combine transcripts from all scenes to create content
            transcripts = []
            for scene in scenes:
                transcript = scene.get("transcript", "")
                if transcript:
                    transcripts.append(transcript)

            # Join all transcripts as the video content
            content = " ".join(transcripts) if transcripts else ""

            # Extract features from first scene frame for objects and tags
            first_scene = scenes[0]
            features = await self.service_clients.extract_features(media_path)
            objects = features.get("objects", [])
            tags = features.get("tags", [])

            # Step 3: Process faces from video
            persons_created, persons_updated, person_ids = await self.process_faces(
                user_id=user_id,
                media_id=media_id,
                media_path=media_path,
                is_video=True
            )

            # Step 4: Generate fused embedding
            embedding = await self.service_clients.embed_video(scenes)

            # Step 5: Upsert to vector database with video timestamps
            start_timestamp = scenes[0].get("start_time", 0.0)
            end_timestamp = scenes[-1].get("end_time", 0.0)

            await self.service_clients.upsert_to_vector_db(
                user_id=user_id,
                media_id=media_id,
                embedding=embedding,
                timestamp=timestamp,
                modality="video",
                objects=objects,
                content=content,
                tags=tags,
                location=location,
                people=person_ids,
                start_timestamp_video=start_timestamp,
                end_timestamp_video=end_timestamp
            )

            logger.info(f"Video pipeline completed successfully for media: {media_id}")

            return {
                "success": True,
                "media_id": media_id,
                "message": f"Video processed successfully ({len(scenes)} scenes)",
                "persons_created": persons_created,
                "persons_updated": persons_updated,
                "embedding_dimension": len(embedding)
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
