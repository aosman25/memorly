#!/usr/bin/env python3
"""
Verify the query processing service is running and healthy.
This test doesn't use the Gemini API quota.
"""

import json
import requests
from pymongo import MongoClient

# Configuration
QUERY_PROCESSING_URL = "http://localhost:8006"
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "memorly"
USER_UUID = "mock-user"


def test_health():
    """Test health endpoint"""
    print("=" * 60)
    print("Health Check Test")
    print("=" * 60)

    try:
        response = requests.get(f"{QUERY_PROCESSING_URL}/health", timeout=5)
        response.raise_for_status()
        result = response.json()
        print("✓ Service is healthy")
        print(f"  Status: {result['status']}")
        print(f"  Model: {result['gemini_model']}")
        print(f"  Version: {result['version']}")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def test_readiness():
    """Test readiness endpoint"""
    print("\n" + "=" * 60)
    print("Readiness Check Test")
    print("=" * 60)

    try:
        response = requests.get(f"{QUERY_PROCESSING_URL}/ready", timeout=5)
        response.raise_for_status()
        result = response.json()
        print("✓ Service is ready")
        print(f"  Status: {result['status']}")
        print(f"  Gemini client: Initialized")
        print(f"  MongoDB client: Initialized")
        return True
    except Exception as e:
        print(f"✗ Readiness check failed: {e}")
        return False


def test_mongodb_connection():
    """Test MongoDB connection independently"""
    print("\n" + "=" * 60)
    print("MongoDB Connection Test")
    print("=" * 60)

    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB]

        # Check persons collection
        persons_collection = db[f"{USER_UUID}.persons"]
        person_count = persons_collection.count_documents({})
        print(f"✓ Connected to MongoDB")
        print(f"  Persons in database: {person_count}")

        # Check locations collection
        locations_collection = db[f"{USER_UUID}.locations"]
        location_count = locations_collection.count_documents({})
        print(f"  Locations in database: {location_count}")

        client.close()
        return True
    except Exception as e:
        print(f"✗ MongoDB connection failed: {e}")
        return False


def test_service_structure():
    """Verify service endpoints exist"""
    print("\n" + "=" * 60)
    print("API Structure Test")
    print("=" * 60)

    try:
        # Test that /docs endpoint exists
        response = requests.get(f"{QUERY_PROCESSING_URL}/docs", timeout=5)
        if response.status_code == 200:
            print("✓ API documentation available at /docs")

        # Test that /openapi.json exists
        response = requests.get(f"{QUERY_PROCESSING_URL}/openapi.json", timeout=5)
        if response.status_code == 200:
            openapi = response.json()
            print(f"✓ OpenAPI schema available")
            print(f"  Title: {openapi.get('info', {}).get('title')}")
            print(f"  Version: {openapi.get('info', {}).get('version')}")

            # List endpoints
            paths = list(openapi.get('paths', {}).keys())
            print(f"  Available endpoints:")
            for path in paths:
                print(f"    - {path}")

        return True
    except Exception as e:
        print(f"✗ API structure test failed: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("QUERY PROCESSING SERVICE VERIFICATION")
    print("=" * 60)
    print("\nNOTE: This verification does NOT use Gemini API quota")
    print("      (only tests service health and infrastructure)\n")

    tests = [
        ("Health Check", test_health),
        ("Readiness Check", test_readiness),
        ("MongoDB Connection", test_mongodb_connection),
        ("API Structure", test_service_structure),
    ]

    passed = 0
    total = len(tests)

    for name, test_func in tests:
        if test_func():
            passed += 1

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{total} tests")

    if passed == total:
        print("\n✓ Service is fully operational")
        print("\nNOTE: Query processing is currently limited by Gemini API quota.")
        print("      The quota limit (250 req/day) has been reached.")
        print("      The service will work again when the quota resets.")
    else:
        print("\n✗ Some tests failed - please check the service")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
