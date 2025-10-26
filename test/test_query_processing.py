#!/usr/bin/env python3
"""
Test script for query processing service.
"""

import json
import requests

# Configuration
QUERY_PROCESSING_URL = "http://localhost:8006"
USER_UUID = "mock-user"

# Test queries
test_queries = [
    "Show me photos with John from our trip to Paris last summer",
    "Find videos of my birthday party with Sarah and Mike",
    "Pictures from the beach at sunset",
    "Show me all images with my laptop and coffee",
    "Videos from New York last winter",
    "Show me memories with Sarah"
]


def test_health():
    """Test health endpoint"""
    print("=" * 60)
    print("Testing Health Check")
    print("=" * 60)

    try:
        response = requests.get(f"{QUERY_PROCESSING_URL}/health", timeout=5)
        response.raise_for_status()
        result = response.json()
        print("✓ Service is healthy")
        print(json.dumps(result, indent=2))
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def process_query(query: str):
    """Process a single query"""
    print(f"\n{'=' * 60}")
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
        print(f"  People IDs: {result['people_ids']}")
        print(f"  Locations: {result['locations']}")
        print(f"  Modalities: {result['modalities']}")

        return True

    except Exception as e:
        print(f"✗ Error processing query: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return False


def main():
    print("=" * 60)
    print("QUERY PROCESSING SERVICE TEST")
    print("=" * 60)
    print()

    # Test health
    if not test_health():
        print("\nService not healthy, exiting")
        return

    # Test queries
    print("\n\n" + "=" * 60)
    print("TESTING QUERIES")
    print("=" * 60)

    success_count = 0
    for query in test_queries:
        if process_query(query):
            success_count += 1
        print(f"\n{'─' * 60}")

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
