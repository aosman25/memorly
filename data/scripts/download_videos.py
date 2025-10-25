#!/usr/bin/env python3
"""
Script to download YouTube videos from dataset.json and cut them based on timestamps.

This script:
1. Reads the dataset.json file
2. Downloads YouTube videos using yt-dlp
3. Cuts videos based on start_time and end_time
4. Organizes videos in subfolders by parent YouTube video ID
5. Saves them with UUID-generated filenames (e.g., yg0y30nWqd0/550e8400-e29b-41d4-a716-446655440000.mp4)
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any
import tempfile
import shutil
import uuid
import re
from urllib.parse import urlparse, parse_qs


def check_dependencies():
    """Check if required dependencies (yt-dlp and ffmpeg) are installed."""
    dependencies = {
        'yt-dlp': 'yt-dlp --version',
        'ffmpeg': 'ffmpeg -version'
    }

    missing = []
    for name, command in dependencies.items():
        try:
            subprocess.run(command.split(), capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing.append(name)

    if missing:
        print(f"Error: Missing dependencies: {', '.join(missing)}")
        print("\nTo install:")
        print("  yt-dlp: pip install yt-dlp")
        print("  ffmpeg: sudo apt-get install ffmpeg (Linux) or brew install ffmpeg (Mac)")
        sys.exit(1)


def load_dataset(json_path: str) -> Dict[str, Any]:
    """Load the dataset from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_youtube_video_id(url: str) -> str:
    """
    Extract the YouTube video ID from a URL.

    Supports formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID

    Args:
        url: YouTube URL

    Returns:
        YouTube video ID (e.g., 'yg0y30nWqd0')
    """
    # Try to match standard youtube.com/watch?v= format
    match = re.search(r'(?:v=|/)([0-9A-Za-z_-]{11})(?:[&\?]|$)', url)
    if match:
        return match.group(1)

    # Try parsing as URL
    parsed = urlparse(url)
    if parsed.hostname in ['www.youtube.com', 'youtube.com']:
        if parsed.path == '/watch':
            query_params = parse_qs(parsed.query)
            if 'v' in query_params:
                return query_params['v'][0]
    elif parsed.hostname == 'youtu.be':
        return parsed.path.lstrip('/')

    # Fallback: use the whole URL as identifier (shouldn't happen)
    return url.replace('/', '_').replace(':', '_').replace('?', '_')


def download_and_cut_video(video_id: str, video_url: str, start_time: float,
                           end_time: float, output_dir: Path, youtube_video_id: str) -> tuple[bool, str, str]:
    """
    Download a YouTube video and cut it to the specified time range.

    Args:
        video_id: The key from the JSON (e.g., '097')
        video_url: YouTube video URL
        start_time: Start time in seconds
        end_time: End time in seconds
        output_dir: Directory to save the processed video
        youtube_video_id: YouTube video ID for creating subfolder

    Returns:
        Tuple of (success: bool, uuid_filename: str, relative_path: str)
    """
    # Create subfolder for this YouTube video
    video_subfolder = output_dir / youtube_video_id
    video_subfolder.mkdir(parents=True, exist_ok=True)

    # Generate UUID for filename
    video_uuid = str(uuid.uuid4())
    output_file = video_subfolder / f"{video_uuid}.mp4"
    relative_path = f"{youtube_video_id}/{video_uuid}.mp4"

    # Note: We don't skip existing files anymore since each run generates new UUIDs

    # Create a temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        temp_video = temp_path / "temp_video.mp4"

        print(f"  Downloading video from {video_url}...")

        # Download the full video using yt-dlp
        download_cmd = [
            'yt-dlp',
            '-f', 'best[ext=mp4]/best',  # Prefer mp4 format
            '-o', str(temp_video),
            '--quiet',
            '--no-warnings',
            video_url
        ]

        try:
            result = subprocess.run(download_cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                print(f"  ✗ Failed to download: {result.stderr}")
                return False, "", ""
        except subprocess.TimeoutExpired:
            print(f"  ✗ Download timed out after 5 minutes")
            return False, "", ""
        except Exception as e:
            print(f"  ✗ Download error: {e}")
            return False, "", ""

        # Check if download was successful
        if not temp_video.exists():
            print(f"  ✗ Download failed - file not found")
            return False, "", ""

        print(f"  Cutting video from {start_time:.2f}s to {end_time:.2f}s...")

        # Calculate duration
        duration = end_time - start_time

        # Cut the video using ffmpeg
        cut_cmd = [
            'ffmpeg',
            '-i', str(temp_video),
            '-ss', str(start_time),
            '-t', str(duration),
            '-c', 'copy',  # Copy without re-encoding for speed
            '-y',  # Overwrite output file if exists
            '-loglevel', 'error',
            str(output_file)
        ]

        try:
            result = subprocess.run(cut_cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                # If copy fails, try re-encoding
                print(f"  Retrying with re-encoding...")
                cut_cmd = [
                    'ffmpeg',
                    '-i', str(temp_video),
                    '-ss', str(start_time),
                    '-t', str(duration),
                    '-c:v', 'libx264',
                    '-c:a', 'aac',
                    '-y',
                    '-loglevel', 'error',
                    str(output_file)
                ]
                result = subprocess.run(cut_cmd, capture_output=True, text=True, timeout=120)
                if result.returncode != 0:
                    print(f"  ✗ Failed to cut video: {result.stderr}")
                    return False, "", ""
        except subprocess.TimeoutExpired:
            print(f"  ✗ Video cutting timed out")
            return False, "", ""
        except Exception as e:
            print(f"  ✗ Cutting error: {e}")
            return False, "", ""

    print(f"  ✓ Successfully created {relative_path}")
    return True, video_uuid, relative_path


def process_dataset(dataset_path: str, output_dir: str, skip_existing: bool = True):
    """
    Process all videos in the dataset.

    Args:
        dataset_path: Path to dataset.json
        output_dir: Directory to save processed videos
        skip_existing: Skip videos that already exist
    """
    print("Loading dataset...")
    dataset = load_dataset(dataset_path)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total = len(dataset)
    successful = 0
    failed = []

    print(f"\nFound {total} videos to process")
    print(f"Output directory: {output_path.absolute()}\n")

    for idx, (video_id, data) in enumerate(dataset.items(), 1):
        print(f"[{idx}/{total}] Processing video {video_id}...")

        try:
            video_url = data['parent_video_id']
            start_time = data['start_time']
            end_time = data['end_time']

            # Extract YouTube video ID for subfolder organization
            youtube_video_id = extract_youtube_video_id(video_url)

            success, video_uuid, relative_path = download_and_cut_video(
                video_id=video_id,
                video_url=video_url,
                start_time=start_time,
                end_time=end_time,
                output_dir=output_path,
                youtube_video_id=youtube_video_id
            )

            if success:
                successful += 1
            else:
                failed.append(video_id)

        except KeyError as e:
            print(f"  ✗ Missing required field: {e}")
            failed.append(video_id)
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            failed.append(video_id)

        print()

    # Print summary
    print("=" * 60)
    print(f"Processing complete!")
    print(f"Successful: {successful}/{total}")
    print(f"Failed: {len(failed)}/{total}")

    if failed:
        print(f"\nFailed video IDs: {', '.join(failed)}")


def main():
    """Main entry point."""
    print("YouTube Video Downloader and Cutter")
    print("=" * 60)

    # Check dependencies
    print("Checking dependencies...")
    check_dependencies()
    print("✓ All dependencies found\n")

    # Set paths
    script_dir = Path(__file__).parent  # data/scripts/
    dataset_path = script_dir.parent / "videos" / "dataset.json"  # data/videos/dataset.json
    output_dir = script_dir.parent / "videos"  # data/videos/

    if not dataset_path.exists():
        print(f"Error: Dataset file not found at {dataset_path}")
        sys.exit(1)

    # Process the dataset
    process_dataset(str(dataset_path), str(output_dir))


if __name__ == "__main__":
    main()
