import base64
import io
import uuid
import requests
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image
import structlog
from deepface import DeepFace
from deepface.modules import detection
import cv2

logger = structlog.get_logger()


def load_image_from_url(url: str) -> np.ndarray:
    """
    Load image from URL and convert to numpy array.

    Args:
        url: URL of the image

    Returns:
        Image as numpy array (BGR format for OpenCV)
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Load image using PIL
        image = Image.open(io.BytesIO(response.content))

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert PIL image to numpy array
        img_array = np.array(image)

        # Convert RGB to BGR (OpenCV format)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        logger.debug("Loaded image from URL", shape=img_bgr.shape)
        return img_bgr

    except Exception as e:
        raise ValueError(f"Failed to load image from URL: {str(e)}")


def load_image_from_base64(base64_str: str) -> np.ndarray:
    """
    Load image from base64 string and convert to numpy array.

    Args:
        base64_str: Base64 encoded image string

    Returns:
        Image as numpy array (BGR format for OpenCV)
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

        # Convert PIL image to numpy array
        img_array = np.array(image)

        # Convert RGB to BGR (OpenCV format)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        logger.debug("Loaded image from base64", shape=img_bgr.shape)
        return img_bgr

    except Exception as e:
        raise ValueError(f"Failed to load image from base64: {str(e)}")


def detect_and_extract_faces(
    image: np.ndarray,
    min_face_size: int = 20,
    detector_backend: str = "retinaface"
) -> List[Tuple[np.ndarray, dict, float]]:
    """
    Detect faces in an image and extract face regions.

    Args:
        image: Input image as numpy array (BGR format)
        min_face_size: Minimum face size in pixels
        detector_backend: Face detector to use (retinaface, mtcnn, opencv, ssd)

    Returns:
        List of tuples: (face_image, face_region_dict, confidence)
    """
    try:
        # Detect faces using DeepFace
        face_objs = DeepFace.extract_faces(
            img_path=image,
            detector_backend=detector_backend,
            enforce_detection=False,
            align=True
        )

        faces = []
        for face_obj in face_objs:
            # Get face region
            facial_area = face_obj.get('facial_area', {})
            confidence = face_obj.get('confidence', 0.0)

            # Skip low confidence detections
            if confidence < 0.5:
                continue

            # Get face coordinates
            x = facial_area.get('x', 0)
            y = facial_area.get('y', 0)
            w = facial_area.get('w', 0)
            h = facial_area.get('h', 0)

            # Skip small faces
            if w < min_face_size or h < min_face_size:
                continue

            # Extract face region from original image
            face_region = image[y:y+h, x:x+w]

            if face_region.size == 0:
                continue

            faces.append((face_region, facial_area, confidence))

        logger.info("Detected faces", count=len(faces))
        return faces

    except Exception as e:
        logger.error("Face detection failed", error=str(e))
        return []


def generate_face_embedding(
    face_image: np.ndarray,
    model_name: str = "ArcFace"
) -> np.ndarray:
    """
    Generate face embedding using ArcFace model.

    Args:
        face_image: Face image as numpy array
        model_name: Model to use for embedding (ArcFace recommended)

    Returns:
        Face embedding as numpy array (512-dimensional for ArcFace)
    """
    try:
        # Generate embedding using DeepFace
        embedding_objs = DeepFace.represent(
            img_path=face_image,
            model_name=model_name,
            enforce_detection=False,
            detector_backend="skip"  # Skip detection as we already have the face
        )

        if not embedding_objs:
            raise ValueError("Failed to generate embedding")

        # Get the embedding vector
        embedding = embedding_objs[0]["embedding"]

        # Convert to numpy array and normalize
        embedding_array = np.array(embedding, dtype=np.float32)

        # Normalize the embedding (L2 normalization)
        norm = np.linalg.norm(embedding_array)
        if norm > 0:
            embedding_array = embedding_array / norm

        logger.debug("Generated face embedding", dimension=len(embedding_array))
        return embedding_array

    except Exception as e:
        logger.error("Embedding generation failed", error=str(e))
        raise


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Cosine similarity score (0-1)
    """
    # Ensure both are normalized
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    # Calculate cosine similarity
    similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

    # Clip to [0, 1] range
    return float(np.clip(similarity, 0.0, 1.0))


def deduplicate_faces(
    faces: List[Tuple[np.ndarray, np.ndarray, float]],
    similarity_threshold: float = 0.4
) -> List[Tuple[np.ndarray, np.ndarray, float, str]]:
    """
    Remove duplicate faces based on embedding similarity.

    Args:
        faces: List of (face_image, embedding, confidence) tuples
        similarity_threshold: Threshold for considering faces as duplicates

    Returns:
        List of unique faces: (face_image, embedding, confidence, face_id)
    """
    if not faces:
        return []

    unique_faces = []
    processed_embeddings = []

    for face_image, embedding, confidence in faces:
        is_duplicate = False

        # Compare with all previously seen faces
        for prev_embedding in processed_embeddings:
            similarity = cosine_similarity(embedding, prev_embedding)

            if similarity >= similarity_threshold:
                is_duplicate = True
                logger.debug("Duplicate face detected", similarity=similarity)
                break

        if not is_duplicate:
            face_id = str(uuid.uuid4())
            unique_faces.append((face_image, embedding, confidence, face_id))
            processed_embeddings.append(embedding)

    logger.info(
        "Face deduplication completed",
        total_faces=len(faces),
        unique_faces=len(unique_faces),
        duplicates_removed=len(faces) - len(unique_faces)
    )

    return unique_faces


def face_image_to_base64(face_image: np.ndarray) -> str:
    """
    Convert face image to base64 string.

    Args:
        face_image: Face image as numpy array (BGR format)

    Returns:
        Base64 encoded image string with data URI prefix
    """
    try:
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(face_rgb)

        # Resize to standard size (for consistency)
        pil_image = pil_image.resize((160, 160), Image.Resampling.LANCZOS)

        # Save to bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=95)
        image_bytes = buffer.getvalue()

        # Encode to base64
        base64_str = base64.b64encode(image_bytes).decode('utf-8')

        return f"data:image/jpeg;base64,{base64_str}"

    except Exception as e:
        raise RuntimeError(f"Failed to encode face image to base64: {str(e)}")


def process_image_for_faces(
    image: np.ndarray,
    min_face_size: int = 20,
    model_name: str = "ArcFace",
    detector_backend: str = "retinaface"
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Process a single image to detect faces and generate embeddings.

    Args:
        image: Input image as numpy array
        min_face_size: Minimum face size in pixels
        model_name: Model to use for embeddings
        detector_backend: Face detector to use

    Returns:
        List of (face_image, embedding, confidence) tuples
    """
    faces_with_embeddings = []

    # Detect faces
    detected_faces = detect_and_extract_faces(image, min_face_size, detector_backend)

    # Generate embeddings for each face
    for face_image, facial_area, confidence in detected_faces:
        try:
            embedding = generate_face_embedding(face_image, model_name)
            faces_with_embeddings.append((face_image, embedding, confidence))
        except Exception as e:
            logger.warning("Failed to generate embedding for face", error=str(e))
            continue

    return faces_with_embeddings


def check_deepface_models() -> bool:
    """
    Check if DeepFace models are available.

    Returns:
        True if models can be loaded, False otherwise
    """
    try:
        # Try to load ArcFace model (will download if not present)
        from deepface.basemodels import ArcFace
        model = ArcFace.loadModel()
        return model is not None
    except Exception as e:
        logger.error("Failed to load DeepFace models", error=str(e))
        return False
