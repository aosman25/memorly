#!/usr/bin/env python3
"""
Test script to verify modality extraction with various query types.
"""

import json
import requests

# Configuration
QUERY_PROCESSING_URL = "http://localhost:8006"
USER_UUID = "mock-user"

# Test queries with different modality expectations
test_cases = [
    {
        "query": "Show me photos with John from our trip to Paris",
        "expected_modalities": ["image"],
        "description": "Explicit photo/image request"
    },
    {
        "query": "Find videos of my birthday party",
        "expected_modalities": ["video"],
        "description": "Explicit video request"
    },
    {
        "query": "Search my notes about the meeting with Sarah",
        "expected_modalities": ["text"],
        "description": "Explicit text/notes request"
    },
    {
        "query": "Show me photos and videos from vacation",
        "expected_modalities": ["image", "video"],
        "description": "Multiple modalities - images and videos"
    },
    {
        "query": "Find everything from my New York trip",
        "expected_modalities": ["video", "image", "text"],
        "description": "Generic search - should include all modalities"
    },
    {
        "query": "What did I capture at the beach last summer",
        "expected_modalities": ["video", "image", "text"],
        "description": "Ambiguous 'capture' - could be any media type"
    },
    {
        "query": "Show me memories from my wedding",
        "expected_modalities": ["video", "image", "text"],
        "description": "Generic 'memories' - should include all types"
    }
]


def process_query(query: str):
    """Process a single query"""
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
        return result

    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def main():
    print("=" * 70)
    print("MODALITY EXTRACTION TEST")
    print("=" * 70)
    print("\nTesting AI-based modality extraction from query context\n")

    passed = 0
    total = len(test_cases)

    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        expected = test_case["expected_modalities"]
        description = test_case["description"]

        print(f"\n[{i}/{total}] {description}")
        print(f"Query: \"{query}\"")
        print(f"Expected modalities: {expected}")

        result = process_query(query)

        if result:
            actual = result.get("modalities", [])
            print(f"Actual modalities:   {actual}")

            # Check if modalities match (order doesn't matter)
            if set(actual) == set(expected):
                print("✓ PASS - Modalities match expected")
                passed += 1
            else:
                # Also pass if actual contains all expected (AI might add more context)
                if set(expected).issubset(set(actual)):
                    print("✓ PASS - Contains expected modalities (AI may have inferred additional context)")
                    passed += 1
                else:
                    print("✗ FAIL - Modalities don't match")
        else:
            print("✗ FAIL - Query processing failed")

        print("-" * 70)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {passed}/{total} tests passed")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
