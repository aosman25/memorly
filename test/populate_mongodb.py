#!/usr/bin/env python3
"""
MongoDB Population Script

This script populates a local MongoDB instance with mock data for testing.
The database structure uses a user UUID as the main collection, with three
subcollections: media, locations, and persons.

User UUID: mock-user
"""

import json
import os
import sys
from pathlib import Path
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

# Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DATABASE_NAME = os.getenv("DATABASE_NAME", "memorly")
USER_UUID = "mock-user"

# Paths to data files
DATA_DIR = Path(__file__).parent.parent / "data"
MEDIA_FILE = DATA_DIR / "media.json"
LOCATIONS_FILE = DATA_DIR / "locations.json"
PERSONS_FILE = DATA_DIR / "persons.json"


def connect_to_mongodb(uri: str, max_retries: int = 5) -> MongoClient:
    """
    Connect to MongoDB with retry logic.

    Args:
        uri: MongoDB connection URI
        max_retries: Maximum number of connection attempts

    Returns:
        MongoClient instance

    Raises:
        ConnectionFailure: If connection fails after max retries
    """
    print(f"Connecting to MongoDB at {uri}...")

    for attempt in range(1, max_retries + 1):
        try:
            client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            # Test the connection
            client.admin.command('ping')
            print(f"✓ Successfully connected to MongoDB")
            return client
        except ConnectionFailure as e:
            if attempt < max_retries:
                print(f"  Connection attempt {attempt}/{max_retries} failed. Retrying...")
            else:
                print(f"✗ Failed to connect to MongoDB after {max_retries} attempts")
                raise


def load_json_file(file_path: Path) -> dict:
    """
    Load and parse a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Parsed JSON data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    print(f"Loading {file_path.name}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"  ✓ Loaded {file_path.name}")
    return data


def populate_media(db, media_data: dict) -> int:
    """
    Populate the media subcollection.

    Args:
        db: MongoDB database instance
        media_data: Dictionary of media objects keyed by ID

    Returns:
        Number of documents inserted
    """
    collection_name = f"{USER_UUID}.media"
    collection = db[collection_name]

    # Clear existing data
    result = collection.delete_many({})
    if result.deleted_count > 0:
        print(f"  Cleared {result.deleted_count} existing documents")

    # Convert dict to list of documents
    documents = list(media_data.values())

    if not documents:
        print(f"  ⚠ No media data to insert")
        return 0

    # Insert documents
    result = collection.insert_many(documents)
    count = len(result.inserted_ids)

    print(f"  ✓ Inserted {count} media documents")

    # Create indexes
    collection.create_index("id", unique=True)
    collection.create_index("timestamp")
    collection.create_index("mediaType")
    collection.create_index("location")

    print(f"  ✓ Created indexes on media collection")

    return count


def populate_locations(db, locations_data: list) -> int:
    """
    Populate the locations subcollection.

    Args:
        db: MongoDB database instance
        locations_data: List of location strings

    Returns:
        Number of documents inserted
    """
    collection_name = f"{USER_UUID}.locations"
    collection = db[collection_name]

    # Clear existing data
    result = collection.delete_many({})
    if result.deleted_count > 0:
        print(f"  Cleared {result.deleted_count} existing documents")

    if not locations_data:
        print(f"  ⚠ No locations data to insert")
        return 0

    # Convert list to documents with proper structure
    documents = [{"location": loc} for loc in locations_data]

    # Insert documents
    result = collection.insert_many(documents)
    count = len(result.inserted_ids)

    print(f"  ✓ Inserted {count} location documents")

    # Create index
    collection.create_index("location", unique=True)

    print(f"  ✓ Created index on locations collection")

    return count


def populate_persons(db, persons_data: dict) -> int:
    """
    Populate the persons subcollection.

    Args:
        db: MongoDB database instance
        persons_data: Dictionary of person objects keyed by ID

    Returns:
        Number of documents inserted
    """
    collection_name = f"{USER_UUID}.persons"
    collection = db[collection_name]

    # Clear existing data
    result = collection.delete_many({})
    if result.deleted_count > 0:
        print(f"  Cleared {result.deleted_count} existing documents")

    # Convert dict to list of documents
    documents = list(persons_data.values())

    if not documents:
        print(f"  ⚠ No persons data to insert")
        return 0

    # Insert documents
    result = collection.insert_many(documents)
    count = len(result.inserted_ids)

    print(f"  ✓ Inserted {count} person documents")

    # Create indexes
    collection.create_index("id", unique=True)
    collection.create_index("name")
    collection.create_index("associated-media")

    print(f"  ✓ Created indexes on persons collection")

    return count


def verify_collections(db) -> None:
    """
    Verify that collections were created and populated correctly.

    Args:
        db: MongoDB database instance
    """
    print("\nVerifying collections...")

    collections = {
        f"{USER_UUID}.media": "Media",
        f"{USER_UUID}.locations": "Locations",
        f"{USER_UUID}.persons": "Persons"
    }

    for collection_name, display_name in collections.items():
        if collection_name in db.list_collection_names():
            count = db[collection_name].count_documents({})
            print(f"  ✓ {display_name}: {count} documents")

            # Show sample document
            sample = db[collection_name].find_one({}, {"_id": 0})
            if sample:
                # Limit display for readability
                sample_str = str(sample)
                if len(sample_str) > 200:
                    sample_str = sample_str[:200] + "..."
                print(f"    Sample: {sample_str}")
        else:
            print(f"  ✗ {display_name}: Collection not found")


def main():
    """Main execution function."""
    print("=" * 60)
    print("MongoDB Population Script")
    print("=" * 60)
    print(f"Database: {DATABASE_NAME}")
    print(f"User UUID: {USER_UUID}")
    print("=" * 60)

    try:
        # Connect to MongoDB
        client = connect_to_mongodb(MONGO_URI)
        db = client[DATABASE_NAME]

        # Load data files
        print("\nLoading data files...")
        media_data = load_json_file(MEDIA_FILE)
        locations_data = load_json_file(LOCATIONS_FILE)
        persons_data = load_json_file(PERSONS_FILE)

        # Populate collections
        print("\nPopulating collections...")
        print(f"\n1. Media Collection ({USER_UUID}.media):")
        media_count = populate_media(db, media_data)

        print(f"\n2. Locations Collection ({USER_UUID}.locations):")
        locations_count = populate_locations(db, locations_data)

        print(f"\n3. Persons Collection ({USER_UUID}.persons):")
        persons_count = populate_persons(db, persons_data)

        # Verify collections
        verify_collections(db)

        # Summary
        print("\n" + "=" * 60)
        print("Summary:")
        print(f"  Total media documents: {media_count}")
        print(f"  Total locations: {locations_count}")
        print(f"  Total persons: {persons_count}")
        print(f"  Total documents: {media_count + locations_count + persons_count}")
        print("=" * 60)
        print("✓ Database population completed successfully!")
        print("=" * 60)

        # Close connection
        client.close()
        return 0

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("Make sure the data files exist in the data/ directory")
        return 1

    except ConnectionFailure as e:
        print(f"\n✗ MongoDB Connection Error: {e}")
        print("Make sure MongoDB is running on", MONGO_URI)
        print("You can start MongoDB using: docker-compose up -d")
        return 1

    except Exception as e:
        print(f"\n✗ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
