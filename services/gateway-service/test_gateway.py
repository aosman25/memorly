#!/usr/bin/env python3
"""
Simple test script for the Gateway Service.
"""
import requests
import time
import sys


def test_health():
    """Test the health endpoint."""
    print("Testing health endpoint...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_process_image(image_path: str):
    """Test image processing."""
    print(f"\nTesting image processing with: {image_path}")
    try:
        with open(image_path, "rb") as f:
            files = {"file": f}
            data = {
                "media_id": f"test-img-{int(time.time())}",
                "user_id": "mock-user",
                "timestamp": int(time.time()),
                "location": "Test Location"
            }
            response = requests.post(
                "http://localhost:8000/process/image",
                files=files,
                data=data,
                timeout=300
            )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_process_text():
    """Test text processing."""
    print("\nTesting text processing...")
    try:
        data = {
            "text": "This is a test memory from my trip to New York City.",
            "media_id": f"test-txt-{int(time.time())}",
            "user_id": "mock-user",
            "timestamp": int(time.time()),
            "location": "New York, NY"
        }
        response = requests.post(
            "http://localhost:8000/process/text",
            data=data,
            timeout=60
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Run tests."""
    print("=" * 60)
    print("Gateway Service Test Suite")
    print("=" * 60)

    # Test 1: Health Check
    health_ok = test_health()

    if not health_ok:
        print("\n❌ Health check failed. Make sure the service is running.")
        print("Run: python main.py")
        sys.exit(1)

    print("\n✓ Health check passed")

    # Test 2: Text Processing (simplest)
    text_ok = test_process_text()
    if text_ok:
        print("\n✓ Text processing passed")
    else:
        print("\n❌ Text processing failed")

    # Test 3: Image Processing (if image path provided)
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        image_ok = test_process_image(image_path)
        if image_ok:
            print("\n✓ Image processing passed")
        else:
            print("\n❌ Image processing failed")
    else:
        print("\nSkipping image test (no image path provided)")
        print("Usage: python test_gateway.py [path/to/image.jpg]")

    print("\n" + "=" * 60)
    print("Test suite completed")
    print("=" * 60)


if __name__ == "__main__":
    main()
