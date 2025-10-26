import os
from typing import List, Tuple
from pymongo import MongoClient
import logging

logger = logging.getLogger(__name__)


def load_prompt(prompt_file: str = "prompt.txt") -> str:
    """
    Load the query extraction prompt from a file.

    Args:
        prompt_file: Path to the prompt file (default: prompt.txt)

    Returns:
        The prompt text as a string
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(current_dir, prompt_file)

    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    except Exception as e:
        raise Exception(f"Error loading prompt file: {str(e)}")


def create_extraction_prompt(query: str) -> str:
    """
    Create the prompt for query feature extraction.

    Args:
        query: The user's search query

    Returns:
        Complete prompt string with the query
    """
    base_prompt = load_prompt()
    return f"{base_prompt}\n\nQuery: \"{query}\""


class MongoDBClient:
    """MongoDB client for person and location lookups"""

    def __init__(self, uri: str, db_name: str):
        """Initialize MongoDB client."""
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        logger.info(f"Connected to MongoDB at {uri}, database: {db_name}")

    def get_persons_collection(self, user_id: str):
        """Get the persons collection for a specific user."""
        collection_name = f"{user_id}.persons"
        return self.db[collection_name]

    def get_locations_collection(self, user_id: str):
        """Get the locations collection for a specific user."""
        collection_name = f"{user_id}.locations"
        return self.db[collection_name]

    def match_person_names(self, user_id: str, names: List[str]) -> List[str]:
        """
        Match person names to person IDs in MongoDB.

        Args:
            user_id: User UUID
            names: List of person names to match

        Returns:
            List of matched person IDs
        """
        if not names:
            return []

        collection = self.get_persons_collection(user_id)
        person_ids = []

        for name in names:
            # Case-insensitive search for person names
            # Match against the "name" field in the persons collection
            person = collection.find_one(
                {"name": {"$regex": f"^{name}$", "$options": "i"}},
                {"id": 1}
            )

            if person:
                person_id = person.get("id")
                if person_id:
                    person_ids.append(person_id)
                    logger.info(f"Matched person name '{name}' to ID: {person_id}")
            else:
                logger.debug(f"No match found for person name: {name}")

        return person_ids

    def match_location_names(self, user_id: str, locations: List[str]) -> List[str]:
        """
        Match location names to location strings in MongoDB.

        Args:
            user_id: User UUID
            locations: List of location names to match

        Returns:
            List of matched location strings (e.g., "New York, NY")
        """
        if not locations:
            return []

        collection = self.get_locations_collection(user_id)
        matched_locations = []

        for location in locations:
            # Case-insensitive partial match for location strings
            # Match against the "location" field
            loc_doc = collection.find_one(
                {"location": {"$regex": location, "$options": "i"}},
                {"location": 1}
            )

            if loc_doc:
                location_str = loc_doc.get("location")
                if location_str and location_str not in matched_locations:
                    matched_locations.append(location_str)
                    logger.info(f"Matched location '{location}' to: {location_str}")
            else:
                logger.debug(f"No match found for location: {location}")

        return matched_locations

    def close(self):
        """Close the MongoDB connection."""
        self.client.close()
        logger.info("Closed MongoDB connection")
