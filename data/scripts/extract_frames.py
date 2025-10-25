#!/usr/bin/env python3
"""
Script to extract random frames from videos and save them as PNG images.

This script:
1. Scans the data/videos directory for video files
2. Groups videos by parent YouTube video ID
3. For each parent YouTube video, extracts 6-9 random frames total across all clips
4. Saves frames as PNG images directly in data/images/{youtube_video_id}/ folder
5. Uses ffmpeg for frame extraction
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List
import random
import re
import uuid


def check_dependencies():
    """Check if required dependencies (ffmpeg) are installed."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg is not installed")
        print("\nTo install:")
        print("  Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("  macOS: brew install ffmpeg")
        sys.exit(1)


def get_video_duration(video_path: Path) -> float:
    """
    Get the duration of a video file in seconds using ffprobe.

    Args:
        video_path: Path to the video file

    Returns:
        Duration in seconds
    """
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"  ✗ Failed to get video duration: {e}")
        return 0.0


def extract_frame(video_path: Path, timestamp: float, output_path: Path) -> bool:
    """
    Extract a single frame from a video at the specified timestamp.

    Args:
        video_path: Path to the video file
        timestamp: Time in seconds to extract the frame
        output_path: Path to save the extracted frame

    Returns:
        True if successful, False otherwise
    """
    cmd = [
        'ffmpeg',
        '-ss', str(timestamp),
        '-i', str(video_path),
        '-frames:v', '1',
        '-q:v', '2',  # High quality
        '-y',  # Overwrite if exists
        '-loglevel', 'error',
        str(output_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"  ✗ Failed to extract frame: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  ✗ Frame extraction timed out")
        return False
    except Exception as e:
        print(f"  ✗ Frame extraction error: {e}")
        return False


def extract_random_frames_from_video(video_path: Path, output_dir: Path, timestamp: float) -> tuple[bool, str]:
    """
    Extract a single random frame from a video.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frame
        timestamp: Timestamp to extract the frame from

    Returns:
        Tuple of (success: bool, uuid_filename: str)
    """
    # Generate UUID for filename
    frame_uuid = str(uuid.uuid4())
    output_file = output_dir / f"{frame_uuid}.png"

    print(f"    Extracting frame at {timestamp:.2f}s from {video_path.name}...")

    if extract_frame(video_path, timestamp, output_file):
        print(f"    ✓ Saved to {output_file.name}")
        return True, frame_uuid
    else:
        print(f"    ✗ Failed to extract frame")
        return False, ""


def process_videos(videos_dir: Path, images_dir: Path):
    """
    Process all videos in the videos directory and extract frames.
    Groups videos by parent YouTube video ID and extracts 6-9 frames total per parent video.

    Args:
        videos_dir: Path to videos directory
        images_dir: Path to images directory
    """
    # Find all video files (organized in subfolders by YouTube ID)
    video_files = list(videos_dir.glob("*/*.mp4"))

    if not video_files:
        print("No video files found in the videos directory")
        return

    # Group videos by YouTube video ID
    videos_by_youtube_id = {}
    for video_path in video_files:
        youtube_video_id = video_path.parent.name
        if youtube_video_id not in videos_by_youtube_id:
            videos_by_youtube_id[youtube_video_id] = []
        videos_by_youtube_id[youtube_video_id].append(video_path)

    total_youtube_videos = len(videos_by_youtube_id)
    total_clips = len(video_files)
    successful_youtube_videos = 0
    total_frames = 0

    print(f"\nFound {total_clips} video clips across {total_youtube_videos} parent YouTube videos")
    print(f"Videos directory: {videos_dir.absolute()}")
    print(f"Images directory: {images_dir.absolute()}\n")

    for idx, (youtube_video_id, clips) in enumerate(videos_by_youtube_id.items(), 1):
        print(f"[{idx}/{total_youtube_videos}] Processing YouTube video: {youtube_video_id}")
        print(f"  Found {len(clips)} clip(s)")

        # Create output directory: images/{youtube_video_id}/
        output_dir = images_dir / youtube_video_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine number of frames to extract for this parent video (6-9 random)
        num_frames_total = random.randint(6, 9)
        print(f"  Extracting {num_frames_total} total frames from all clips...")

        # Randomly select which clips to extract from and how many frames from each
        frames_extracted = 0

        # Create a list of (video_path, duration) tuples
        video_durations = []
        for clip in clips:
            duration = get_video_duration(clip)
            if duration > 0:
                video_durations.append((clip, duration))

        if not video_durations:
            print(f"  ✗ No valid clips found\n")
            continue

        # Distribute frames across clips (randomly select clips with replacement)
        for _ in range(num_frames_total):
            # Randomly select a clip
            clip_path, duration = random.choice(video_durations)

            # Generate random timestamp (avoid first and last 0.5 seconds)
            if duration > 1.0:
                timestamp = random.uniform(0.5, duration - 0.5)
            else:
                timestamp = duration / 2.0

            # Extract the frame
            success, frame_uuid = extract_random_frames_from_video(clip_path, output_dir, timestamp)
            if success:
                frames_extracted += 1

        if frames_extracted > 0:
            successful_youtube_videos += 1
            total_frames += frames_extracted
            print(f"  ✓ Extracted {frames_extracted}/{num_frames_total} frames to {youtube_video_id}/\n")
        else:
            print(f"  ✗ Failed to extract any frames\n")

    # Print summary
    print("=" * 60)
    print(f"Processing complete!")
    print(f"Parent YouTube videos processed: {successful_youtube_videos}/{total_youtube_videos}")
    print(f"Total frames extracted: {total_frames}")
    print(f"Average frames per YouTube video: {total_frames/successful_youtube_videos:.1f}" if successful_youtube_videos > 0 else "")


def main():
    """Main entry point."""
    print("Video Frame Extractor")
    print("=" * 60)

    # Check dependencies
    print("Checking dependencies...")
    check_dependencies()
    print("✓ ffmpeg found\n")

    # Set paths
    script_dir = Path(__file__).parent  # data/scripts/
    videos_dir = script_dir.parent / "videos"  # data/videos/
    images_dir = script_dir.parent / "images"  # data/images/

    if not videos_dir.exists():
        print(f"Error: Videos directory not found at {videos_dir}")
        sys.exit(1)

    # Create images directory if it doesn't exist
    images_dir.mkdir(parents=True, exist_ok=True)

    # Process all videos
    process_videos(videos_dir, images_dir)


if __name__ == "__main__":
    main()
