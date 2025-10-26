import numpy as np
from pymongo import MongoClient, ASCENDING
from typing import List, Optional, Dict, Tuple
import logging
from config import Config

logger = logging.getLogger(__name__)


class MongoDBClient:
    def __init__(self, uri: str, db_name: str):
        """Initialize MongoDB client."""
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        logger.info(f"Connected to MongoDB at {uri}, database: {db_name}")

    def get_persons_collection(self, user_id: str):
        """Get the persons collection for a specific user."""
        collection_name = f"{user_id}.persons"
        return self.db[collection_name]

    def cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)

        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = np.dot(emb1, emb2) / (norm1 * norm2)
        return float(np.clip(similarity, 0.0, 1.0))

    def find_matching_person(
        self,
        user_id: str,
        face_embedding: List[float],
        threshold: float = 0.4
    ) -> Optional[str]:
        """
        Find a person in the database whose face embedding matches the given embedding.

        Args:
            user_id: User UUID
            face_embedding: Face embedding to match
            threshold: Cosine similarity threshold for matching

        Returns:
            Person ID if match found, None otherwise
        """
        collection = self.get_persons_collection(user_id)

        # Get all persons with embeddings
        persons = collection.find({"embedding": {"$exists": True, "$ne": None}})

        for person in persons:
            person_embedding = person.get("embedding")
            if not person_embedding:
                continue

            similarity = self.cosine_similarity(face_embedding, person_embedding)
            logger.debug(f"Comparing with person {person['id']}: similarity={similarity:.4f}")

            if similarity >= threshold:
                logger.info(f"Found matching person: {person['id']} (similarity={similarity:.4f})")
                return person["id"]

        return None

    def create_person(
        self,
        user_id: str,
        person_id: str,
        face_embedding: List[float],
        media_id: str
    ) -> bool:
        """
        Create a new person in the database.

        Args:
            user_id: User UUID
            person_id: New person ID
            face_embedding: Face embedding
            media_id: ID of the media where this face was found

        Returns:
            True if successful
        """
        collection = self.get_persons_collection(user_id)

        person_doc = {
            "id": person_id,
            "name": None,
            "relationship": None,
            "associated-media": [media_id],
            "embedding": face_embedding
        }

        result = collection.insert_one(person_doc)
        logger.info(f"Created new person: {person_id} with media: {media_id}")
        return result.acknowledged

    def add_media_to_person(
        self,
        user_id: str,
        person_id: str,
        media_id: str
    ) -> bool:
        """
        Add a media ID to a person's associated-media list.

        Args:
            user_id: User UUID
            person_id: Person ID
            media_id: Media ID to add

        Returns:
            True if successful
        """
        collection = self.get_persons_collection(user_id)

        result = collection.update_one(
            {"id": person_id},
            {"$addToSet": {"associated-media": media_id}}
        )

        if result.modified_count > 0:
            logger.info(f"Added media {media_id} to person {person_id}")
        else:
            logger.debug(f"Media {media_id} already associated with person {person_id}")

        return result.acknowledged

    def get_all_persons(self, user_id: str) -> List[Dict]:
        """Get all persons for a user."""
        collection = self.get_persons_collection(user_id)
        return list(collection.find({}))

    def close(self):
        """Close the MongoDB connection."""
        self.client.close()
        logger.info("Closed MongoDB connection")
