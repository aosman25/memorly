#!/usr/bin/env python3
"""
Script to populate Milvus vector database with mock media data.

This script:
1. Reads media.json for metadata
2. Finds corresponding media files (images or videos)
3. Sends media to gateway service for processing
4. Tracks success/failure rates
"""

import argparse
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
VIDEOS_DIR = DATA_DIR / "videos"

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


def find_media_file(media_id: str, file_format: str, media_type: str) -> Path:
    """Find the media file for a given media ID."""
    # Media files are organized in subdirectories
    # Search recursively for the file
    pattern = f"{media_id}.{file_format}"

    search_dir = IMAGES_DIR if media_type == "image" else VIDEOS_DIR

    for file_path in search_dir.rglob(pattern):
        return file_path

    return None


def process_media(media_id: str, metadata: Dict, media_path: Path, media_type: str) -> Tuple[bool, str]:
    """
    Send media (image or video) to gateway for processing.

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Prepare form data
        with open(media_path, 'rb') as f:
            # Determine content type and endpoint
            if media_type == "image":
                content_type = f'image/{metadata["fileFormat"]}'
                endpoint = f"{GATEWAY_URL}/process/image"
            else:  # video
                content_type = f'video/{metadata["fileFormat"]}'
                endpoint = f"{GATEWAY_URL}/process/video"

            files = {
                'file': (media_path.name, f, content_type)
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
                endpoint,
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


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Populate Milvus vector database with mock media data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process only images (default)
  python populate_vector_db.py

  # Process only videos
  python populate_vector_db.py --videos-only

  # Process both images and videos
  python populate_vector_db.py --all

  # Process specific media types
  python populate_vector_db.py --media-type video
        """
    )

    parser.add_argument(
        '--videos-only',
        action='store_true',
        help='Process only video files'
    )
    parser.add_argument(
        '--images-only',
        action='store_true',
        help='Process only image files (default)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process both images and videos'
    )
    parser.add_argument(
        '--media-type',
        choices=['image', 'video'],
        help='Specify media type to process (alternative to --videos-only/--images-only)'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    # Determine which media types to process
    if args.all:
        media_types = ['image', 'video']
    elif args.videos_only or args.media_type == 'video':
        media_types = ['video']
    else:
        # Default: images only
        media_types = ['image']

    print("=" * 60)
    print("VECTOR DATABASE POPULATION SCRIPT")
    print("=" * 60)
    print(f"Gateway URL: {GATEWAY_URL}")
    print(f"User UUID:   {USER_UUID}")
    print(f"Data Dir:    {DATA_DIR}")
    print(f"Media Types: {', '.join(media_types)}")
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

    # Filter for selected media types
    media_items = {
        k: v for k, v in media_data.items()
        if v.get('mediaType') in media_types
    }
    stats['total'] = len(media_items)

    media_type_str = "media files" if len(media_types) > 1 else f"{media_types[0]}s"
    print(f"\nFound {len(media_items)} {media_type_str} to process")
    print("=" * 60)

    # Ask for confirmation
    response = input(f"\nProcess {len(media_items)} {media_type_str}? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        sys.exit(0)

    print("\nStarting processing...\n")
    stats['start_time'] = time.time()

    # Process each media item
    for idx, (media_id, metadata) in enumerate(media_items.items(), 1):
        media_type = metadata['mediaType']
        type_icon = "🖼️ " if media_type == "image" else "🎥"

        print(f"[{idx}/{len(media_items)}] {type_icon} {media_id[:8]}... ({metadata.get('location', 'N/A')})")

        # Find media file
        media_path = find_media_file(media_id, metadata['fileFormat'], media_type)

        if not media_path:
            print(f"  ⚠ {media_type.capitalize()} file not found, skipping")
            stats['skipped'] += 1
            continue

        print(f"  📁 {media_path.relative_to(DATA_DIR)}")

        # Process the media
        success, message = process_media(media_id, metadata, media_path, media_type)

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
        print(f"\n✓ All {media_type_str} processed successfully!")
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
