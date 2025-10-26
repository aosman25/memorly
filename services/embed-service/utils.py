import base64
import io
import numpy as np
from typing import List, Optional, Tuple
from PIL import Image
import structlog

logger = structlog.get_logger()


def decode_base64_image(base64_str: str) -> Image.Image:
    """
    Decode base64 string to PIL Image.

    Args:
        base64_str: Base64 encoded image string

    Returns:
        PIL Image object
    """
    try:
        # Remove data URI prefix if present
        if base64_str.startswith('data:'):
            base64_str = base64_str.split(',', 1)[1]

        # Decode base64 string
        image_bytes = base64.b64decode(base64_str)

        # Load image using PIL
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        logger.debug("Decoded base64 image", size=image.size, mode=image.mode)
        return image

    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {str(e)}")


def image_to_base64_for_api(image: Image.Image, max_size: int = 512) -> str:
    """
    Convert PIL Image to base64 string for API submission.
    Resizes image to fit within max_size while maintaining aspect ratio.

    Args:
        image: PIL Image object
        max_size: Maximum dimension (width or height) for the resized image

    Returns:
        Base64 encoded string suitable for API
    """
    try:
        # Resize image to reduce size for API
        # Calculate new size maintaining aspect ratio
        width, height = image.size
        if width > max_size or height > max_size:
            if width > height:
                new_width = max_size
                new_height = int((max_size / width) * height)
            else:
                new_height = max_size
                new_width = int((max_size / height) * width)

            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")

        # Convert image to bytes
        buffer = io.BytesIO()
        image.save(buffer, format='PNG', optimize=True)
        image_bytes = buffer.getvalue()

        # Encode to base64
        base64_str = base64.b64encode(image_bytes).decode('utf-8')

        # Return with data URI prefix for API
        return f"data:image/png;base64,{base64_str}"

    except Exception as e:
        raise RuntimeError(f"Failed to convert image to base64: {str(e)}")


def embed_image(
    image_base64: str,
    client,
    model: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
) -> np.ndarray:
    """
    Generate embedding for an image using CLIP model.

    Args:
        image_base64: Base64 encoded image
        client: OpenAI client instance
        model: Model name for embedding

    Returns:
        512-dimensional embedding as numpy array
    """
    try:
        # Decode the base64 image
        image = decode_base64_image(image_base64)

        # Resize and re-encode to reduce size (256x256 to fit API limits)
        image_base64 = image_to_base64_for_api(image, max_size=256)

        embeddings = client.embeddings.create(
            model=model,
            input=image_base64,
            encoding_format="float"
        )

        embedding = embeddings.data[0].embedding
        embedding_array = np.array(embedding, dtype=np.float32)

        logger.debug("Generated image embedding", dimension=len(embedding_array))
        return embedding_array

    except Exception as e:
        logger.error("Image embedding failed", error=str(e))
        raise RuntimeError(f"Failed to generate image embedding: {str(e)}")


def embed_text(
    text: str,
    client,
    model: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
) -> np.ndarray:
    """
    Generate embedding for text using CLIP model.

    Args:
        text: Input text
        client: OpenAI client instance
        model: Model name for embedding

    Returns:
        512-dimensional embedding as numpy array
    """
    try:
        if not text or not text.strip():
            # Return zero vector for empty text
            logger.warning("Empty text provided, returning zero vector")
            return np.zeros(512, dtype=np.float32)

        embeddings = client.embeddings.create(
            model=model,
            input=text.strip(),
            encoding_format="float"
        )

        embedding = embeddings.data[0].embedding
        embedding_array = np.array(embedding, dtype=np.float32)

        logger.debug("Generated text embedding", dimension=len(embedding_array), text_length=len(text))
        return embedding_array

    except Exception as e:
        logger.error("Text embedding failed", error=str(e))
        raise RuntimeError(f"Failed to generate text embedding: {str(e)}")


def fuse_embeddings(
    visual_embedding: np.ndarray,
    text_embedding: np.ndarray,
    alpha: float = 0.6,
    beta: float = 0.4
) -> np.ndarray:
    """
    Fuse visual and text embeddings using weighted combination.

    Args:
        visual_embedding: Visual embedding vector
        text_embedding: Text embedding vector
        alpha: Weight for visual embedding (default: 0.6)
        beta: Weight for text embedding (default: 0.4)

    Returns:
        Fused embedding vector
    """
    try:
        # Ensure both embeddings have the same dimension
        if visual_embedding.shape != text_embedding.shape:
            raise ValueError(
                f"Embedding dimensions mismatch: visual={visual_embedding.shape}, "
                f"text={text_embedding.shape}"
            )

        # Normalize weights to sum to 1.0
        total_weight = alpha + beta
        if total_weight > 0:
            alpha = alpha / total_weight
            beta = beta / total_weight

        # Fuse embeddings
        fused = (alpha * visual_embedding + beta * text_embedding)

        # Normalize the fused embedding
        norm = np.linalg.norm(fused)
        if norm > 0:
            fused = fused / norm

        logger.debug(
            "Fused embeddings",
            alpha=alpha,
            beta=beta,
            dimension=len(fused)
        )

        return fused

    except Exception as e:
        logger.error("Embedding fusion failed", error=str(e))
        raise RuntimeError(f"Failed to fuse embeddings: {str(e)}")


def embed_video_frames(
    frames: List[Tuple[str, Optional[str]]],
    client,
    alpha: float = 0.6,
    beta: float = 0.4,
    model: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
) -> np.ndarray:
    """
    Embed multiple video frames and fuse them into a single embedding.

    For each frame:
    - If transcript is available: fuse visual and text embeddings
    - If no transcript: use only visual embedding

    Then average all frame embeddings to get final video embedding.

    Args:
        frames: List of (frame_base64, transcript) tuples
        client: OpenAI client instance
        alpha: Weight for visual embeddings
        beta: Weight for text embeddings
        model: Model name for embedding

    Returns:
        512-dimensional fused video embedding
    """
    try:
        frame_embeddings = []

        for i, (frame_base64, transcript) in enumerate(frames, 1):
            logger.info(f"Processing frame {i}/{len(frames)}")

            # Generate visual embedding
            visual_emb = embed_image(frame_base64, client, model)

            # Generate and fuse with text embedding if transcript is available
            if transcript and transcript.strip():
                text_emb = embed_text(transcript, client, model)
                frame_emb = fuse_embeddings(visual_emb, text_emb, alpha, beta)
            else:
                # No transcript, use visual embedding only
                frame_emb = visual_emb
                # Normalize
                norm = np.linalg.norm(frame_emb)
                if norm > 0:
                    frame_emb = frame_emb / norm

            frame_embeddings.append(frame_emb)

        # Average all frame embeddings
        video_embedding = np.mean(frame_embeddings, axis=0)

        # Normalize the final embedding
        norm = np.linalg.norm(video_embedding)
        if norm > 0:
            video_embedding = video_embedding / norm

        logger.info(
            "Video embedding completed",
            total_frames=len(frames),
            dimension=len(video_embedding)
        )

        return video_embedding

    except Exception as e:
        logger.error("Video embedding failed", error=str(e))
        raise RuntimeError(f"Failed to generate video embedding: {str(e)}")


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    Normalize embedding to unit length.

    Args:
        embedding: Input embedding vector

    Returns:
        Normalized embedding vector
    """
    norm = np.linalg.norm(embedding)
    if norm > 0:
        return embedding / norm
    return embedding
