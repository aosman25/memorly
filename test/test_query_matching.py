#!/usr/bin/env python3
"""
Test script to verify person and location matching in query processing.
"""

import json
import requests
from pymongo import MongoClient

# Configuration
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "memorly"
QUERY_PROCESSING_URL = "http://localhost:8006"
USER_UUID = "mock-user"


def setup_test_data():
    """Add test persons and locations to MongoDB"""
    print("=" * 60)
    print("Setting up test data in MongoDB")
    print("=" * 60)

    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]

    # Add test persons
    persons_collection = db[f"{USER_UUID}.persons"]
    test_persons = [
        {
            "id": "person-john-123",
            "name": "John",
            "relationship": None,
            "associated-media": [],
            "embedding": [0.1] * 512  # Dummy embedding
        },
        {
            "id": "person-sarah-456",
            "name": "Sarah",
            "relationship": None,
            "associated-media": [],
            "embedding": [0.2] * 512
        },
        {
            "id": "person-mike-789",
            "name": "Mike",
            "relationship": None,
            "associated-media": [],
            "embedding": [0.3] * 512
        }
    ]

    # Clear existing test persons
    persons_collection.delete_many({"name": {"$in": ["John", "Sarah", "Mike"]}})
    persons_collection.insert_many(test_persons)
    print(f"✓ Added {len(test_persons)} test persons")

    # Add test locations
    locations_collection = db[f"{USER_UUID}.locations"]
    test_locations = [
        {
            "id": "location-newyork-002",
            "name": "New York",
            "location": "New York, USA",
            "country": "USA"
        },
        {
            "id": "location-beach-003",
            "name": "beach",
            "location": "Beach Location",
            "type": "generic"
        }
    ]

    # Clear existing test locations (skip Paris since it already exists)
    locations_collection.delete_many({"name": {"$in": ["New York", "beach"]}})
    locations_collection.insert_many(test_locations)
    print(f"✓ Added {len(test_locations)} test locations (Paris already exists)")

    client.close()
    print()


def test_query_with_matching(query: str):
    """Test a query and show matching results"""
    print(f"{'=' * 60}")
    print(f"Query: {query}")
    print(f"{'=' * 60}\n")

    try:
        data = {
            "query": query,
            "user_id": USER_UUID
        }

        response = requests.post(
            f"{QUERY_PROCESSING_URL}/process-query",
            json=data,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()

        print(f"✓ Query processed successfully\n")
        print(f"Processing time: {result['processing_time_ms']:.2f}ms")
        print(f"\nExtracted Information:")
        print(f"  Objects: {result['objects']}")
        print(f"  Tags: {result['tags']}")
        print(f"  Content: {result['content']}")
        print(f"  People Names Extracted: {', '.join(result.get('people_names', [])) if result.get('people_names') else 'None'}")
        print(f"  People IDs Matched: {result['people_ids']}")
        print(f"  Location Names Extracted: {', '.join(result.get('location_names', [])) if result.get('location_names') else 'None'}")
        print(f"  Location IDs Matched: {result['location_ids']}")

        return True

    except Exception as e:
        print(f"✗ Error processing query: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return False


def cleanup_test_data():
    """Remove test data from MongoDB"""
    print("\n" + "=" * 60)
    print("Cleaning up test data")
    print("=" * 60)

    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]

    # Remove test persons
    persons_collection = db[f"{USER_UUID}.persons"]
    result = persons_collection.delete_many({"name": {"$in": ["John", "Sarah", "Mike"]}})
    print(f"✓ Removed {result.deleted_count} test persons")

    # Remove test locations
    locations_collection = db[f"{USER_UUID}.locations"]
    result = locations_collection.delete_many({"name": {"$in": ["Paris", "New York", "beach"]}})
    print(f"✓ Removed {result.deleted_count} test locations")

    client.close()


def main():
    print("=" * 60)
    print("QUERY PROCESSING MATCHING TEST")
    print("=" * 60)
    print()

    # Setup test data
    setup_test_data()

    # Test queries with known persons and locations
    test_queries = [
        "Show me photos with John from our trip to Paris last summer",
        "Find videos of my birthday party with Sarah and Mike",
        "Pictures from the beach at sunset",
        "Show me all images from New York"
    ]

    success_count = 0
    for query in test_queries:
        if test_query_with_matching(query):
            success_count += 1
        print(f"\n{'─' * 60}\n")

    # Cleanup
    cleanup_test_data()

    # Summary
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Processed: {success_count}/{len(test_queries)} queries successfully")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
