#!/usr/bin/env python3
"""
Script to generate media.json file aggregating all media information.

This script:
1. Scans all images in data/images/
2. Scans all videos in data/videos/
3. Scans all journals in data/journals/
4. Creates a unified media.json file with metadata for each item
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import subprocess
import random
from datetime import datetime, timedelta


def get_file_timestamp(file_path: Path) -> int:
    """Get the creation/modification timestamp of a file."""
    return int(file_path.stat().st_mtime)


def get_video_timestamp(video_path: Path) -> int:
    """
    Try to extract video creation timestamp from metadata using ffprobe.
    Falls back to file modification time if ffprobe is not available.
    """
    try:
        # Try to get creation time from video metadata using ffprobe
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries',
             'format_tags=creation_time', '-of', 'json', str(video_path)],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            data = json.loads(result.stdout)
            creation_time = data.get('format', {}).get('tags', {}).get('creation_time')
            if creation_time:
                from datetime import datetime
                dt = datetime.fromisoformat(creation_time.replace('Z', '+00:00'))
                return int(dt.timestamp())
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass

    # Fallback to file modification time
    return get_file_timestamp(video_path)


def collect_images(images_dir: Path) -> List[Dict]:
    """Collect all image files and their metadata."""
    images = []

    # Group images by parent folder
    folders = {}
    for image_path in images_dir.glob('*/*.png'):
        parent_folder = image_path.parent.name
        if parent_folder not in folders:
            folders[parent_folder] = []
        folders[parent_folder].append(image_path)

    # Assign timestamps - same day, few minutes apart for same folder
    for parent_folder, image_paths in folders.items():
        # Generate a random base date within the past year
        days_ago = random.randint(0, 365)
        base_date = datetime.now() - timedelta(days=days_ago)
        # Set to a random time during the day
        base_date = base_date.replace(
            hour=random.randint(8, 20),
            minute=random.randint(0, 59),
            second=random.randint(0, 59)
        )
        base_timestamp = int(base_date.timestamp())

        # Assign timestamps with few minutes apart
        for idx, image_path in enumerate(image_paths):
            image_id = image_path.stem
            # Add 1-5 minutes between each image in the same folder
            minutes_offset = idx * random.randint(1, 5) * 60
            timestamp = base_timestamp + minutes_offset

            images.append({
                'id': image_id,
                'timestamp': timestamp,
                'fileFormat': 'png',
                'mediaType': 'image'
            })

    return images


def collect_videos(videos_dir: Path) -> List[Dict]:
    """Collect all video files and their metadata."""
    videos = []

    # Group videos by parent folder
    folders = {}
    for video_path in videos_dir.glob('*/*.mp4'):
        parent_folder = video_path.parent.name
        if parent_folder not in folders:
            folders[parent_folder] = []
        folders[parent_folder].append(video_path)

    # Assign timestamps - same day, few minutes apart for same folder
    for parent_folder, video_paths in folders.items():
        # Generate a random base date within the past year
        days_ago = random.randint(0, 365)
        base_date = datetime.now() - timedelta(days=days_ago)
        # Set to a random time during the day
        base_date = base_date.replace(
            hour=random.randint(8, 20),
            minute=random.randint(0, 59),
            second=random.randint(0, 59)
        )
        base_timestamp = int(base_date.timestamp())

        # Assign timestamps with few minutes apart
        for idx, video_path in enumerate(video_paths):
            video_id = video_path.stem
            # Add 1-5 minutes between each video in the same folder
            minutes_offset = idx * random.randint(1, 5) * 60
            timestamp = base_timestamp + minutes_offset

            videos.append({
                'id': video_id,
                'timestamp': timestamp,
                'fileFormat': 'mp4',
                'mediaType': 'video'
            })

    return videos


def collect_journals(journals_dir: Path) -> List[Dict]:
    """Collect all journal files and their metadata."""
    journals = []

    # Find all .txt journal files
    for journal_path in journals_dir.glob('*.txt'):
        journal_id = journal_path.stem
        # Generate random timestamp for journals
        days_ago = random.randint(0, 365)
        base_date = datetime.now() - timedelta(days=days_ago)
        base_date = base_date.replace(
            hour=random.randint(8, 22),
            minute=random.randint(0, 59),
            second=random.randint(0, 59)
        )
        timestamp = int(base_date.timestamp())

        journals.append({
            'id': journal_id,
            'timestamp': timestamp,
            'fileFormat': 'txt',
            'mediaType': 'journal'
        })

    return journals


def generate_media_json(data_dir: Path, output_path: Path):
    """Generate the unified media.json file."""
    print("Generating media.json...")
    print("=" * 60)

    images_dir = data_dir / "images"
    videos_dir = data_dir / "videos"
    journals_dir = data_dir / "journals"

    # Collect all media
    print("\nCollecting images...")
    images = collect_images(images_dir) if images_dir.exists() else []
    print(f"  Found {len(images)} images")

    print("\nCollecting videos...")
    videos = collect_videos(videos_dir) if videos_dir.exists() else []
    print(f"  Found {len(videos)} videos")

    print("\nCollecting journals...")
    journals = collect_journals(journals_dir) if journals_dir.exists() else []
    print(f"  Found {len(journals)} journals")

    # Combine all media
    all_media = images + videos + journals

    # Convert to dictionary with id as key
    media_dict = {item['id']: item for item in all_media}

    # Save to file
    print(f"\nSaving media.json...")
    with open(output_path, 'w') as f:
        json.dump(media_dict, f, indent=2)

    print("=" * 60)
    print(f"✓ Successfully generated media.json")
    print(f"  Total media items: {len(media_dict)}")
    print(f"    - Images: {len(images)}")
    print(f"    - Videos: {len(videos)}")
    print(f"    - Journals: {len(journals)}")
    print(f"\nOutput: {output_path}")


def main():
    """Main entry point."""
    print("Media JSON Generator")
    print("=" * 60)

    # Set paths
    script_dir = Path(__file__).parent  # data/scripts/
    data_dir = script_dir.parent  # data/
    output_path = data_dir / "media.json"

    # Generate media.json
    generate_media_json(data_dir, output_path)


if __name__ == "__main__":
    main()
