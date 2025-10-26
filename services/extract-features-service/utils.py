import base64
import requests
import os
from typing import Optional
from google.genai import types


def load_prompt(prompt_file: str = "prompt.txt") -> str:
    """
    Load the feature extraction prompt from a file.

    Args:
        prompt_file: Path to the prompt file (default: prompt.txt)

    Returns:
        The prompt text as a string
    """
    # Get the directory where this file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(current_dir, prompt_file)

    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    except Exception as e:
        raise Exception(f"Error loading prompt file: {str(e)}")


def load_image_from_url(url: str) -> types.Part:
    """
    Load image from URL and convert to Gemini Part object.

    Args:
        url: URL of the image

    Returns:
        types.Part object containing image data
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Determine MIME type from content-type header
        content_type = response.headers.get('content-type', 'image/jpeg')

        return types.Part.from_bytes(
            data=response.content,
            mime_type=content_type
        )
    except Exception as e:
        raise ValueError(f"Failed to load image from URL: {str(e)}")


def load_image_from_base64(base64_str: str, mime_type: str = "image/jpeg") -> types.Part:
    """
    Load image from base64 string and convert to Gemini Part object.

    Args:
        base64_str: Base64 encoded image string (can include data URI prefix or not)
        mime_type: MIME type of the image (default: image/jpeg)

    Returns:
        types.Part object containing image data
    """
    try:
        # Remove data URI prefix if present
        if base64_str.startswith('data:'):
            # Extract MIME type from data URI
            header, base64_str = base64_str.split(',', 1)
            if 'image/' in header:
                mime_type = header.split(';')[0].split(':')[1]

        # Decode base64 string
        image_bytes = base64.b64decode(base64_str)

        return types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type
        )
    except Exception as e:
        raise ValueError(f"Failed to load image from base64: {str(e)}")


def create_extraction_prompt() -> list:
    """
    Create the prompt for feature extraction.

    Returns:
        List containing the prompt text loaded from file
    """
    return [load_prompt()]
