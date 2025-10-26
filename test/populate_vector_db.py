#!/usr/bin/env python3
"""
Script to populate Milvus vector database with mock image data.

This script:
1. Reads media.json for metadata
2. Finds corresponding image files
3. Sends images to gateway service for processing
4. Tracks success/failure rates
"""

import json
import os
import sys
import time
import requests
from pathlib import Path
from typing import Dict, List, Tuple

# Configuration
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:9000")
USER_UUID = "mock-user"
DATA_DIR = Path(__file__).parent.parent / "data"
MEDIA_JSON = DATA_DIR / "media.json"
IMAGES_DIR = DATA_DIR / "images"

# Statistics
stats = {
    "total": 0,
    "processed": 0,
    "failed": 0,
    "skipped": 0,
    "start_time": None,
    "end_time": None
}


def load_media_metadata() -> Dict:
    """Load media.json metadata."""
    print(f"Loading media metadata from: {MEDIA_JSON}")

    if not MEDIA_JSON.exists():
        print(f"❌ Error: {MEDIA_JSON} not found")
        sys.exit(1)

    with open(MEDIA_JSON, 'r') as f:
        media_data = json.load(f)

    print(f"✓ Loaded {len(media_data)} media entries")
    return media_data


def find_image_file(media_id: str, file_format: str) -> Path:
    """Find the image file for a given media ID."""
    # Images are organized in subdirectories
    # Search recursively for the file
    pattern = f"{media_id}.{file_format}"

    for img_path in IMAGES_DIR.rglob(pattern):
        return img_path

    return None


def process_image(media_id: str, metadata: Dict, image_path: Path) -> Tuple[bool, str]:
    """
    Send image to gateway for processing.

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Prepare form data
        with open(image_path, 'rb') as f:
            files = {
                'file': (image_path.name, f, f'image/{metadata["fileFormat"]}')
            }

            data = {
                'media_id': media_id,
                'user_id': USER_UUID,
                'timestamp': metadata['timestamp'],
                'location': metadata.get('location', '')
            }

            # Send to gateway
            print(f"  → Sending to gateway...", end='', flush=True)
            response = requests.post(
                f"{GATEWAY_URL}/process/image",
                files=files,
                data=data,
                timeout=300  # 5 minutes timeout
            )

            response.raise_for_status()
            result = response.json()

            if result.get('success'):
                persons_info = f"(persons: +{result.get('persons_created', 0)}/~{result.get('persons_updated', 0)})"
                print(f" ✓ {persons_info}")
                return True, result.get('message', 'Success')
            else:
                print(f" ✗")
                return False, result.get('message', 'Unknown error')

    except requests.exceptions.Timeout:
        print(f" ✗ TIMEOUT")
        return False, "Request timeout (>5 minutes)"
    except requests.exceptions.RequestException as e:
        print(f" ✗ ERROR")
        return False, f"Request error: {str(e)}"
    except Exception as e:
        print(f" ✗ EXCEPTION")
        return False, f"Exception: {str(e)}"


def check_gateway_health() -> bool:
    """Check if gateway is healthy."""
    print("Checking gateway health...")
    try:
        response = requests.get(f"{GATEWAY_URL}/health", timeout=5)
        health = response.json()

        if health.get('status') == 'healthy':
            print("✓ Gateway is healthy")
            services = health.get('services', {})
            for service, status in services.items():
                icon = "✓" if status else "✗"
                print(f"  {icon} {service}")
            return True
        else:
            print(f"⚠ Gateway status: {health.get('status')}")
            return False

    except Exception as e:
        print(f"❌ Gateway not accessible: {e}")
        return False


def print_statistics():
    """Print processing statistics."""
    duration = stats['end_time'] - stats['start_time']

    print("\n" + "=" * 60)
    print("PROCESSING STATISTICS")
    print("=" * 60)
    print(f"Total images:      {stats['total']}")
    print(f"Successfully processed: {stats['processed']} ({stats['processed']/stats['total']*100:.1f}%)")
    print(f"Failed:            {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")
    print(f"Skipped:           {stats['skipped']} ({stats['skipped']/stats['total']*100:.1f}%)")
    print(f"Duration:          {duration:.1f} seconds")
    print(f"Average:           {duration/stats['total']:.2f} seconds per image")
    print("=" * 60)


def main():
    """Main execution function."""
    print("=" * 60)
    print("VECTOR DATABASE POPULATION SCRIPT")
    print("=" * 60)
    print(f"Gateway URL: {GATEWAY_URL}")
    print(f"User UUID:   {USER_UUID}")
    print(f"Data Dir:    {DATA_DIR}")
    print("=" * 60)
    print()

    # Check gateway health
    if not check_gateway_health():
        print("\n❌ Gateway is not healthy. Please start the services first:")
        print("   docker-compose up -d")
        sys.exit(1)

    print()

    # Load metadata
    media_data = load_media_metadata()

    # Filter for images only
    images = {k: v for k, v in media_data.items() if v.get('mediaType') == 'image'}
    stats['total'] = len(images)

    print(f"\nFound {len(images)} images to process")
    print("=" * 60)

    # Ask for confirmation
    response = input(f"\nProcess {len(images)} images? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        sys.exit(0)

    print("\nStarting processing...\n")
    stats['start_time'] = time.time()

    # Process each image
    for idx, (media_id, metadata) in enumerate(images.items(), 1):
        print(f"[{idx}/{len(images)}] {media_id[:8]}... ({metadata.get('location', 'N/A')})")

        # Find image file
        image_path = find_image_file(media_id, metadata['fileFormat'])

        if not image_path:
            print(f"  ⚠ Image file not found, skipping")
            stats['skipped'] += 1
            continue

        print(f"  📁 {image_path.relative_to(DATA_DIR)}")

        # Process the image
        success, message = process_image(media_id, metadata, image_path)

        if success:
            stats['processed'] += 1
        else:
            stats['failed'] += 1
            print(f"  ⚠ Failure reason: {message}")

        # Small delay to avoid overwhelming the services
        time.sleep(0.5)

    stats['end_time'] = time.time()

    # Print statistics
    print_statistics()

    # Exit code based on results
    if stats['failed'] > 0:
        print(f"\n⚠ Completed with {stats['failed']} failures")
        sys.exit(1)
    else:
        print(f"\n✓ All images processed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        if stats.get('start_time'):
            stats['end_time'] = time.time()
            print_statistics()
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
