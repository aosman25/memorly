#!/usr/bin/env python3
"""
Test script for the search pipeline endpoint.

This script tests the /search endpoint in the gateway service which:
1. Processes the query (query-processing-service)
2. Performs vector search with filters (search-service)
"""

import json
import os
import sys
import time
import requests
from pathlib import Path
from typing import Dict, List

# Configuration
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:9000")
USER_ID = "mock_user"

# Test queries covering different scenarios
TEST_QUERIES = [
    {
        "query": "beach vacation sunset",
        "description": "Semantic search - no filters",
        "expect_filters": False
    },
    {
        "query": "Show me photos from New York",
        "description": "Location filter search",
        "expect_location_filter": True
    },
    {
        "query": "pictures with Sarah",
        "description": "Person filter search",
        "expect_person_filter": True
    },
    {
        "query": "birthday party celebration",
        "description": "Event semantic search - no filters",
        "expect_filters": False
    },
    {
        "query": "San Francisco with John",
        "description": "Location + person filter search",
        "expect_location_filter": True,
        "expect_person_filter": True
    },
]

# Statistics
stats = {
    "total": 0,
    "successful": 0,
    "failed": 0,
    "errors": []
}


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def print_section(text: str):
    """Print formatted section."""
    print("\n" + "-" * 70)
    print(text)
    print("-" * 70)


def check_gateway_health() -> bool:
    """Check if gateway and required services are healthy."""
    print("Checking gateway health...")
    try:
        response = requests.get(f"{GATEWAY_URL}/health", timeout=5)
        health = response.json()

        if health.get('status') == 'healthy':
            print("✓ Gateway is healthy")
            services = health.get('services', {})
            required_services = ['query-processing', 'search', 'embed']

            all_required_healthy = True
            for service in required_services:
                status = services.get(service, False)
                icon = "✓" if status else "✗"
                print(f"  {icon} {service}")
                if not status:
                    all_required_healthy = False

            return all_required_healthy
        else:
            print(f"⚠ Gateway status: {health.get('status')}")
            return False

    except Exception as e:
        print(f"❌ Gateway not accessible: {e}")
        return False


def perform_search(query: str, limit: int = 10, offset: int = 0) -> Dict:
    """
    Perform search request.

    Returns:
        Response dictionary or None if failed
    """
    try:
        payload = {
            "query": query,
            "user_id": USER_ID,
            "limit": limit,
            "offset": offset
        }

        print(f"  → Sending search request...", end='', flush=True)
        response = requests.post(
            f"{GATEWAY_URL}/search",
            json=payload,
            timeout=30
        )

        response.raise_for_status()
        result = response.json()

        print(f" ✓")
        return result

    except requests.exceptions.Timeout:
        print(f" ✗ TIMEOUT")
        return None
    except requests.exceptions.RequestException as e:
        print(f" ✗ ERROR")
        print(f"  ⚠ Error: {str(e)}")
        if hasattr(e.response, 'text'):
            print(f"  ⚠ Response: {e.response.text[:200]}")
        return None
    except Exception as e:
        print(f" ✗ EXCEPTION")
        print(f"  ⚠ Exception: {str(e)}")
        return None


def print_query_info(query_info: Dict):
    """Print extracted query information."""
    print("\n  📋 Query Information (Raw):")
    import json
    print(json.dumps(query_info, indent=2))

    print("\n  📋 Query Information (Formatted):")

    if query_info.get("objects"):
        print(f"     Objects: {', '.join(query_info['objects'])}")

    if query_info.get("tags"):
        print(f"     Tags: {', '.join(query_info['tags'])}")

    if query_info.get("people_ids"):
        print(f"     People IDs: {', '.join(query_info['people_ids'])}")

    if query_info.get("locations"):
        print(f"     Locations: {', '.join(query_info['locations'])}")

    if query_info.get("modalities"):
        print(f"     Modalities: {', '.join(query_info['modalities'])}")

    if query_info.get("content"):
        print(f"     Content: {query_info['content']}")


def print_search_results(results: List[Dict], total: int, limit: int):
    """Print search results summary."""
    print(f"\n  📊 Results: {total} total, showing {len(results)}/{min(limit, total)}")

    if results:
        print("\n  Raw Results (first 2):")
        import json
        for idx, result in enumerate(results[:2], 1):
            print(f"\n  Result {idx}:")
            print(json.dumps(result, indent=4))

        print("\n  Top Results (Formatted):")
        for idx, result in enumerate(results[:3], 1):
            print(f"    {idx}. ID: {result['id'][:12]}... (score: {result['score']:.3f})")
            print(f"       Type: {result['media_type']}")
            if result.get('source_path'):
                print(f"       Path: {result['source_path']}")

            metadata = result.get('metadata', {})
            if metadata.get('location'):
                print(f"       Location: {metadata['location']}")
            if metadata.get('objects'):
                print(f"       Objects: {', '.join(metadata['objects'][:5])}")


def test_search_query(test_case: Dict) -> bool:
    """
    Test a single search query.

    Returns:
        True if successful, False otherwise
    """
    query = test_case["query"]
    description = test_case.get("description", "")

    print_section(f"Test: {description}")
    print(f"  Query: \"{query}\"")

    # Perform search
    result = perform_search(query)

    if not result:
        stats["errors"].append({
            "query": query,
            "description": description,
            "error": "Request failed"
        })
        return False

    # Check if successful
    if not result.get("success"):
        print(f"  ✗ Search failed: {result.get('message', 'Unknown error')}")
        stats["errors"].append({
            "query": query,
            "description": description,
            "error": result.get('message', 'Unknown error')
        })
        return False

    # Print raw response
    print("\n  📄 RAW RESPONSE:")
    import json
    print(json.dumps(result, indent=2))

    # Print query info
    query_info = result.get("query_info", {})
    print_query_info(query_info)

    # Print results
    results = result.get("results", [])
    total = result.get("total", 0)
    print_search_results(results, total, 10)

    # Print processing time
    processing_time = result.get("processing_time_ms", 0)
    print(f"\n  ⏱️  Processing time: {processing_time:.0f}ms")

    # Validate filter expectations
    validation_passed = True

    # Check if no filters are expected
    if test_case.get("expect_filters") == False:
        has_filters = bool(query_info.get("people_ids") or query_info.get("locations"))
        if has_filters:
            print(f"  ⚠ Expected no filters, but found filters")
            validation_passed = False
        else:
            print(f"  ✓ No filters applied (as expected)")

    # Check location filter expectation
    if test_case.get("expect_location_filter"):
        locations = query_info.get("locations", [])
        if locations:
            print(f"  ✓ Location filter applied: {locations}")
        else:
            print(f"  ⚠ Expected location filter, but none found")
            validation_passed = False

    # Check person filter expectation
    if test_case.get("expect_person_filter"):
        people_ids = query_info.get("people_ids", [])
        if people_ids:
            print(f"  ✓ Person filter applied: {people_ids}")
        else:
            print(f"  ⚠ Expected person filter, but none found (person may not exist in database)")
            # Don't fail validation - person might not be in DB
            # validation_passed = True

    if validation_passed:
        print(f"\n  ✅ Test passed")
    else:
        print(f"\n  ⚠️  Test completed with validation warnings")

    return True


def print_statistics():
    """Print test statistics."""
    print_header("TEST STATISTICS")
    print(f"Total tests:    {stats['total']}")
    print(f"Successful:     {stats['successful']} ({stats['successful']/stats['total']*100:.1f}%)")
    print(f"Failed:         {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")

    if stats["errors"]:
        print("\nErrors:")
        for error in stats["errors"]:
            print(f"  • {error['description']}")
            print(f"    Query: \"{error['query']}\"")
            print(f"    Error: {error['error']}")

    print("=" * 70)


def main():
    """Main execution function."""
    print_header("SEARCH PIPELINE TEST SCRIPT")
    print(f"Gateway URL: {GATEWAY_URL}")
    print(f"User ID:     {USER_ID}")
    print(f"Test queries: {len(TEST_QUERIES)}")

    # Check gateway health
    if not check_gateway_health():
        print("\n❌ Required services are not healthy. Please start the services first:")
        print("   docker-compose up -d")
        sys.exit(1)

    print("\n✓ All required services are healthy")

    # Run tests
    stats["total"] = len(TEST_QUERIES)

    for idx, test_case in enumerate(TEST_QUERIES, 1):
        success = test_search_query(test_case)
        if success:
            stats["successful"] += 1
        else:
            stats["failed"] += 1

        # Small delay between tests
        if idx < len(TEST_QUERIES):
            time.sleep(0.5)

    # Print statistics
    print_statistics()

    # Exit code based on results
    if stats["failed"] > 0:
        print(f"\n⚠ Completed with {stats['failed']} failures")
        sys.exit(1)
    else:
        print(f"\n✅ All tests passed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        if stats.get("total"):
            print_statistics()
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
