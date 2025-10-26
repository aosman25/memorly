import base64
import subprocess
import tempfile
import os
import json
import requests
from typing import List, Tuple, Optional
from pathlib import Path
import structlog

logger = structlog.get_logger()


def download_video(url: str, output_path: str) -> None:
    """
    Download video from URL to local file.

    Args:
        url: URL of the video
        output_path: Path to save the video
    """
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info("Video downloaded successfully", size_bytes=os.path.getsize(output_path))
    except Exception as e:
        raise ValueError(f"Failed to download video: {str(e)}")


def save_base64_video(base64_data: str, output_path: str) -> None:
    """
    Save base64 encoded video to file.

    Args:
        base64_data: Base64 encoded video string
        output_path: Path to save the video
    """
    try:
        # Remove data URI prefix if present
        if base64_data.startswith('data:'):
            base64_data = base64_data.split(',', 1)[1]

        video_bytes = base64.b64decode(base64_data)

        with open(output_path, 'wb') as f:
            f.write(video_bytes)

        logger.info("Video saved from base64", size_bytes=len(video_bytes))
    except Exception as e:
        raise ValueError(f"Failed to save base64 video: {str(e)}")


def get_video_duration(video_path: str) -> float:
    """
    Get video duration using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        Duration in seconds
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json',
            video_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        duration = float(data['format']['duration'])

        logger.info("Got video duration", duration=duration)
        return duration
    except Exception as e:
        raise RuntimeError(f"Failed to get video duration: {str(e)}")


def detect_scenes(video_path: str, threshold: float = 0.3) -> List[Tuple[float, float]]:
    """
    Detect scene changes using ffmpeg scene detection.

    Args:
        video_path: Path to video file
        threshold: Scene detection threshold (0-1). Lower = more sensitive

    Returns:
        List of (start_time, end_time) tuples for each scene
    """
    try:
        # Get video duration first
        duration = get_video_duration(video_path)

        # Use ffmpeg scene detection filter
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-filter:v', f'select=\'gt(scene,{threshold})\',showinfo',
            '-f', 'null',
            '-'
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            stderr=subprocess.STDOUT
        )

        # Parse scene change timestamps from ffmpeg output
        scene_times = [0.0]  # Always start with 0
        for line in result.stdout.split('\n'):
            if 'pts_time:' in line:
                try:
                    time_str = line.split('pts_time:')[1].split()[0]
                    scene_times.append(float(time_str))
                except (IndexError, ValueError):
                    continue

        # Add final timestamp
        if not scene_times or scene_times[-1] < duration:
            scene_times.append(duration)

        # Remove duplicates and sort
        scene_times = sorted(set(scene_times))

        # Create scene ranges
        scenes = []
        for i in range(len(scene_times) - 1):
            scenes.append((scene_times[i], scene_times[i + 1]))

        logger.info("Detected scenes", scene_count=len(scenes))
        return scenes

    except Exception as e:
        raise RuntimeError(f"Failed to detect scenes: {str(e)}")


def extract_frame(video_path: str, timestamp: float, output_path: str) -> None:
    """
    Extract a single frame from video at specified timestamp.

    Args:
        video_path: Path to video file
        timestamp: Timestamp in seconds
        output_path: Path to save the frame image
    """
    try:
        cmd = [
            'ffmpeg',
            '-ss', str(timestamp),
            '-i', video_path,
            '-frames:v', '1',
            '-q:v', '2',
            '-y',
            output_path
        ]

        subprocess.run(cmd, capture_output=True, check=True)
        logger.debug("Extracted frame", timestamp=timestamp)
    except Exception as e:
        raise RuntimeError(f"Failed to extract frame at {timestamp}s: {str(e)}")


def frame_to_base64(frame_path: str) -> str:
    """
    Convert frame image to base64 string.

    Args:
        frame_path: Path to frame image

    Returns:
        Base64 encoded image string with data URI prefix
    """
    try:
        with open(frame_path, 'rb') as f:
            image_bytes = f.read()

        base64_str = base64.b64encode(image_bytes).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_str}"
    except Exception as e:
        raise RuntimeError(f"Failed to encode frame to base64: {str(e)}")


def extract_audio_segment(video_path: str, start_time: float, end_time: float, output_path: str) -> None:
    """
    Extract audio segment from video.

    Args:
        video_path: Path to video file
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Path to save audio file
    """
    try:
        duration = end_time - start_time
        cmd = [
            'ffmpeg',
            '-ss', str(start_time),
            '-i', video_path,
            '-t', str(duration),
            '-vn',  # No video
            '-acodec', 'libmp3lame',
            '-ar', '16000',  # 16kHz sample rate for Whisper
            '-ac', '1',  # Mono
            '-y',
            output_path
        ]

        subprocess.run(cmd, capture_output=True, check=True)
        logger.debug("Extracted audio segment", start=start_time, end=end_time)
    except Exception as e:
        raise RuntimeError(f"Failed to extract audio segment: {str(e)}")


def transcribe_audio(
    audio_path: str,
    api_key: str,
    base_url: str = "https://api.deepinfra.com/v1/openai",
    language: Optional[str] = None
) -> str:
    """
    Transcribe audio using OpenAI-compatible Whisper API.

    Args:
        audio_path: Path to audio file
        api_key: API key for the service
        base_url: Base URL for the API
        language: Optional language code (ISO-639-1)

    Returns:
        Transcribed text
    """
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        with open(audio_path, 'rb') as audio_file:
            params = {
                "model": "whisper-1",
                "file": audio_file,
            }

            if language:
                params["language"] = language

            transcript = client.audio.transcriptions.create(**params)

        text = transcript.text if hasattr(transcript, 'text') else str(transcript)
        logger.debug("Transcribed audio", text_length=len(text))
        return text

    except Exception as e:
        logger.error("Failed to transcribe audio", error=str(e))
        return ""  # Return empty string on failure rather than crashing


def check_ffmpeg_available() -> bool:
    """
    Check if ffmpeg is available in the system.

    Returns:
        True if ffmpeg is available, False otherwise
    """
    try:
        subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_ffprobe_available() -> bool:
    """
    Check if ffprobe is available in the system.

    Returns:
        True if ffprobe is available, False otherwise
    """
    try:
        subprocess.run(
            ['ffprobe', '-version'],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
