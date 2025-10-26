#!/usr/bin/env python3
"""
Quick test script - processes just 3 images to test the pipeline.
"""

import json
import os
import sys
import time
import requests
from pathlib import Path
from typing import Dict

# Configuration
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:9000")
USER_UUID = "mock-user"
DATA_DIR = Path(__file__).parent.parent / "data"
MEDIA_JSON = DATA_DIR / "media.json"
IMAGES_DIR = DATA_DIR / "images"
MAX_IMAGES = 3  # Only process 3 images for quick test


def check_gateway():
    """Check if gateway is accessible."""
    print("Checking gateway...")
    try:
        response = requests.get(f"{GATEWAY_URL}/health", timeout=5)
        health = response.json()
        if health.get('status') == 'healthy':
            print("✓ Gateway is healthy\n")
            return True
    except Exception as e:
        print(f"❌ Gateway error: {e}\n")
    return False


def find_image_file(media_id: str, file_format: str) -> Path:
    """Find the image file."""
    pattern = f"{media_id}.{file_format}"
    for img_path in IMAGES_DIR.rglob(pattern):
        return img_path
    return None


def process_image(media_id: str, metadata: Dict, image_path: Path):
    """Process a single image."""
    print(f"\n{'='*60}")
    print(f"Processing: {media_id[:8]}...")
    print(f"Location: {metadata.get('location', 'N/A')}")
    print(f"File: {image_path.relative_to(DATA_DIR)}")
    print(f"{'='*60}\n")

    try:
        with open(image_path, 'rb') as f:
            files = {'file': (image_path.name, f, f'image/{metadata["fileFormat"]}')}
            data = {
                'media_id': media_id,
                'user_id': USER_UUID,
                'timestamp': metadata['timestamp'],
                'location': metadata.get('location', '')
            }

            print("→ Sending to gateway...")
            start = time.time()
            response = requests.post(
                f"{GATEWAY_URL}/process/image",
                files=files,
                data=data,
                timeout=180
            )
            duration = time.time() - start

            response.raise_for_status()
            result = response.json()

            print(f"✓ Completed in {duration:.1f}s\n")
            print("Response:")
            print(json.dumps(result, indent=2))

            return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    print("="*60)
    print("QUICK PIPELINE TEST - Processing 3 Images")
    print("="*60)
    print()

    if not check_gateway():
        print("Please start the gateway service first")
        sys.exit(1)

    # Load media
    with open(MEDIA_JSON, 'r') as f:
        media_data = json.load(f)

    # Get first 3 images
    images = {k: v for k, v in media_data.items() if v.get('mediaType') == 'image'}
    images = dict(list(images.items())[:MAX_IMAGES])

    print(f"Testing with {len(images)} images\n")

    success_count = 0
    for idx, (media_id, metadata) in enumerate(images.items(), 1):
        image_path = find_image_file(media_id, metadata['fileFormat'])

        if not image_path:
            print(f"[{idx}/{MAX_IMAGES}] ⚠ Image not found: {media_id}")
            continue

        print(f"[{idx}/{MAX_IMAGES}]")
        if process_image(media_id, metadata, image_path):
            success_count += 1

        if idx < len(images):
            print(f"\n{'─'*60}")
            print("Waiting 2 seconds before next image...")
            print(f"{'─'*60}")
            time.sleep(2)

    print(f"\n{'='*60}")
    print(f"RESULTS: {success_count}/{len(images)} images processed successfully")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
