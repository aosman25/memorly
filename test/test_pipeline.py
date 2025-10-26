#!/usr/bin/env python3
"""
Quick test script - processes 3 images and 2 videos to test the pipeline.
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
VIDEOS_DIR = DATA_DIR / "videos"
NOTES_DIR = DATA_DIR / "notes"
MAX_IMAGES = 5  # Process 5 images
MAX_VIDEOS = 5  # Process 5 videos
MAX_TEXTS = 5   # Process 5 texts


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


def find_video_file(media_id: str, file_format: str) -> Path:
    """Find the video file."""
    pattern = f"{media_id}.{file_format}"
    for vid_path in VIDEOS_DIR.rglob(pattern):
        return vid_path
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


def process_text(media_id: str, metadata: Dict):
    """Process a single text entry."""
    print(f"\n{'='*60}")
    print(f"Processing: {media_id[:8]}...")
    print(f"Location: {metadata.get('location', 'N/A')}")
    print(f"Content preview: {metadata.get('textContent', '')[:50]}...")
    print(f"{'='*60}\n")

    try:
        data = {
            'media_id': media_id,
            'user_id': USER_UUID,
            'timestamp': metadata['timestamp'],
            'location': metadata.get('location', ''),
            'text_content': metadata.get('textContent', '')
        }

        print("→ Sending to gateway...")
        start = time.time()
        response = requests.post(
            f"{GATEWAY_URL}/process/text",
            json=data,
            timeout=60
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


def process_video(media_id: str, metadata: Dict, video_path: Path):
    """Process a single video."""
    print(f"\n{'='*60}")
    print(f"Processing: {media_id[:8]}...")
    print(f"Location: {metadata.get('location', 'N/A')}")
    print(f"File: {video_path.relative_to(DATA_DIR)}")
    print(f"{'='*60}\n")

    try:
        with open(video_path, 'rb') as f:
            files = {'file': (video_path.name, f, f'video/{metadata["fileFormat"]}')}
            data = {
                'media_id': media_id,
                'user_id': USER_UUID,
                'timestamp': metadata['timestamp'],
                'location': metadata.get('location', '')
            }

            print("→ Sending to gateway...")
            start = time.time()
            response = requests.post(
                f"{GATEWAY_URL}/process/video",
                files=files,
                data=data,
                timeout=600  # 10 minutes for long videos
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
    print("COMPREHENSIVE PIPELINE TEST - 5 Images + 5 Videos + 5 Texts")
    print("="*60)
    print()

    if not check_gateway():
        print("Please start the gateway service first")
        sys.exit(1)

    # Load media
    with open(MEDIA_JSON, 'r') as f:
        media_data = json.load(f)

    # Get first 5 images
    images = {k: v for k, v in media_data.items() if v.get('mediaType') == 'image'}
    images = dict(list(images.items())[:MAX_IMAGES])

    # Get first 5 videos
    videos = {k: v for k, v in media_data.items() if v.get('mediaType') == 'video'}
    videos = dict(list(videos.items())[:MAX_VIDEOS])

    # Get first 5 text files from notes directory
    text_files = sorted(list(NOTES_DIR.glob("*.txt")))[:MAX_TEXTS]

    print(f"Testing with {len(images)} images, {len(videos)} videos, and {len(text_files)} texts\n")

    # Process images
    print("="*60)
    print("PROCESSING IMAGES")
    print("="*60)

    image_success = 0
    for idx, (media_id, metadata) in enumerate(images.items(), 1):
        image_path = find_image_file(media_id, metadata['fileFormat'])

        if not image_path:
            print(f"[{idx}/{MAX_IMAGES}] ⚠ Image not found: {media_id}")
            continue

        print(f"[{idx}/{MAX_IMAGES}]")
        if process_image(media_id, metadata, image_path):
            image_success += 1

        if idx < len(images):
            print(f"\n{'─'*60}")
            print("Waiting 2 seconds before next image...")
            print(f"{'─'*60}")
            time.sleep(2)

    # Process videos
    print("\n\n" + "="*60)
    print("PROCESSING VIDEOS")
    print("="*60)

    video_success = 0
    for idx, (media_id, metadata) in enumerate(videos.items(), 1):
        video_path = find_video_file(media_id, metadata['fileFormat'])

        if not video_path:
            print(f"[{idx}/{MAX_VIDEOS}] ⚠ Video not found: {media_id}")
            continue

        print(f"[{idx}/{MAX_VIDEOS}]")
        if process_video(media_id, metadata, video_path):
            video_success += 1

        if idx < len(videos):
            print(f"\n{'─'*60}")
            print("Waiting 3 seconds before next video...")
            print(f"{'─'*60}")
            time.sleep(3)

    # Process texts
    print("\n\n" + "="*60)
    print("PROCESSING TEXTS")
    print("="*60)

    text_success = 0
    for idx, text_file in enumerate(text_files, 1):
        # Read text content from file
        with open(text_file, 'r') as f:
            text_content = f.read()

        # Create metadata for text
        media_id = text_file.stem  # Use filename without extension as ID
        metadata = {
            'timestamp': int(time.time()),
            'location': '',
            'textContent': text_content
        }

        print(f"[{idx}/{MAX_TEXTS}]")
        if process_text(media_id, metadata):
            text_success += 1

        if idx < len(text_files):
            print(f"\n{'─'*60}")
            print("Waiting 1 second before next text...")
            print(f"{'─'*60}")
            time.sleep(1)

    # Final results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Images: {image_success}/{len(images)} processed successfully")
    print(f"Videos: {video_success}/{len(videos)} processed successfully")
    print(f"Texts:  {text_success}/{len(text_files)} processed successfully")
    print(f"Total:  {image_success + video_success + text_success}/{len(images) + len(videos) + len(text_files)} processed successfully")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
