#!/usr/bin/env python3
"""
Test script for LLM Response Generation Service

This script tests the streaming response generation with SSE.
"""

import requests
import json
import sys

# Configuration
LLM_SERVICE_URL = "http://localhost:9000"
USER_ID = "mock_user"

# Test queries
TEST_QUERIES = [
    "Tell me about Christmas celebrations in my memories",
    "What do you know about hair styling activities?",
    "Show me memories from New York"
]


def print_header(text):
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


def test_streaming_response(query: str):
    """Test the streaming response from LLM service"""
    print_header(f"Testing Query: {query}")

    payload = {
        "query": query,
        "user_id": USER_ID,
        "limit": 3
    }

    try:
        # Make streaming request
        print("\n📤 Sending request...")
        with requests.post(
            f"{LLM_SERVICE_URL}/generate",
            json=payload,
            stream=True,
            timeout=60
        ) as response:
            response.raise_for_status()

            print("✓ Connection established, receiving chunks...\n")

            metadata_received = False
            response_text_parts = []

            # Process SSE stream
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')

                    # SSE format: "data: {json}"
                    if line.startswith('data: '):
                        data_str = line[6:]  # Remove "data: " prefix

                        try:
                            chunk = json.loads(data_str)
                            chunk_type = chunk.get('type')

                            if chunk_type == 'metadata':
                                if not metadata_received:
                                    print("📋 METADATA CHUNK:")
                                    print("-" * 80)
                                    sources = chunk['data'].get('sources', [])
                                    print(f"Retrieved {len(sources)} sources:")
                                    for idx, source in enumerate(sources[:3], 1):
                                        print(f"\n  Source {idx}:")
                                        print(f"    ID: {source['id'][:20]}...")
                                        print(f"    Type: {source['media_type']}")
                                        print(f"    Score: {source['score']:.3f}")
                                        metadata = source.get('metadata', {})
                                        if metadata.get('location'):
                                            print(f"    Location: {metadata['location']}")
                                        if metadata.get('tags'):
                                            print(f"    Tags: {', '.join(metadata['tags'][:3])}")
                                    print("-" * 80)
                                    metadata_received = True

                            elif chunk_type == 'response':
                                # Accumulate response text
                                text = chunk.get('data', '')
                                response_text_parts.append(text)
                                # Print without newline for streaming effect
                                print(text, end='', flush=True)

                            elif chunk_type == 'done':
                                print("\n\n✅ GENERATION COMPLETE")
                                processing_time = chunk['data'].get('processing_time_ms', 0)
                                print(f"⏱️  Total processing time: {processing_time:.0f}ms")

                            elif chunk_type == 'error':
                                print(f"\n❌ ERROR: {chunk['data'].get('error')}")

                        except json.JSONDecodeError:
                            continue

            # Show full response
            if response_text_parts:
                print("\n" + "-" * 80)
                print("FULL RESPONSE:")
                print("-" * 80)
                print(''.join(response_text_parts))
                print("-" * 80)

            return True

    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text[:500]}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_health():
    """Check service health"""
    print("Checking LLM Response Service health...")
    try:
        response = requests.get(f"{LLM_SERVICE_URL}/health", timeout=5)
        health = response.json()

        print(f"Status: {health.get('status')}")
        print(f"  Gemini API configured: {health.get('gemini_api_configured')}")
        print(f"  Search service: {health.get('search_service')}")

        return health.get('status') == 'healthy'
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def main():
    print_header("LLM RESPONSE GENERATION SERVICE TEST")
    print(f"Service URL: {LLM_SERVICE_URL}")
    print(f"User ID: {USER_ID}")

    # Health check
    if not check_health():
        print("\n⚠️  Service is not healthy. Please check configuration.")
        sys.exit(1)

    print("\n✓ Service is healthy\n")

    # Run tests
    success_count = 0
    for idx, query in enumerate(TEST_QUERIES, 1):
        if test_streaming_response(query):
            success_count += 1

        # Wait between requests
        if idx < len(TEST_QUERIES):
            input("\nPress Enter to continue to next test...")

    # Summary
    print_header("TEST SUMMARY")
    print(f"Total tests: {len(TEST_QUERIES)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(TEST_QUERIES) - success_count}")
    print("=" * 80)

    if success_count == len(TEST_QUERIES):
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {len(TEST_QUERIES) - success_count} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
