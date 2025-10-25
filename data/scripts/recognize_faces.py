#!/usr/bin/env python3
"""
Script to recognize faces using DeepFace library.

This script:
1. Scans all images in data/images/ directory
2. Detects faces using DeepFace (supports multiple backends)
3. Gets face embeddings using state-of-the-art models (Facenet512, ArcFace, etc.)
4. Matches against existing persons using cosine similarity
5. If similarity >= threshold, matches to existing person
6. If similarity < threshold, creates new person entry
7. Saves headshot images in data/persons/headshots/
8. Updates dataset.json with person information and associated media
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
import uuid
import numpy as np

# Import required libraries
try:
    from deepface import DeepFace
    import cv2
except ImportError as e:
    print(f"Error: Missing required library: {e}")
    print("\nPlease install required packages:")
    print("  pip install deepface opencv-python")
    sys.exit(1)


# Configuration
MODEL_NAME = "Facenet512"  # Options: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, ArcFace, Dlib, SFace
DETECTOR_BACKEND = "opencv"  # Options: opencv, ssd, dlib, mtcnn, retinaface, mediapipe
DISTANCE_METRIC = "cosine"  # Options: cosine, euclidean, euclidean_l2
SIMILARITY_THRESHOLD = 0.4  # Cosine similarity threshold (higher is more similar)

# Name pools for random generation
MALE_NAMES = [
    "James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph",
    "Thomas", "Christopher", "Daniel", "Matthew", "Anthony", "Mark", "Donald", "Steven",
    "Andrew", "Paul", "Joshua", "Kenneth", "Kevin", "Brian", "George", "Timothy"
]

FEMALE_NAMES = [
    "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica",
    "Sarah", "Karen", "Nancy", "Lisa", "Betty", "Margaret", "Sandra", "Ashley",
    "Kimberly", "Emily", "Donna", "Michelle", "Carol", "Amanda", "Melissa", "Deborah"
]

MALE_RELATIONSHIPS = [
    "Father", "Brother", "Grandfather", "Uncle", "Nephew", "Son",
    "Husband", "Father-in-law", "Brother-in-law", "Grandson",
    "Stepfather", "Stepbrother", "Half-brother", "Cousin"
]

FEMALE_RELATIONSHIPS = [
    "Mother", "Sister", "Grandmother", "Aunt", "Niece", "Daughter",
    "Wife", "Mother-in-law", "Sister-in-law", "Granddaughter",
    "Stepmother", "Stepsister", "Half-sister", "Cousin"
]


def load_persons_dataset(dataset_path: Path) -> Dict:
    """Load the persons dataset from JSON file."""
    if dataset_path.exists():
        with open(dataset_path, 'r') as f:
            return json.load(f)
    return {}


def save_persons_dataset(dataset: Dict, dataset_path: Path):
    """Save the persons dataset to JSON file."""
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_path, 'w') as f:
        json.dump(dataset, f, indent=2)


def detect_and_encode_faces(image_path: Path) -> List[Tuple[np.ndarray, List[float], Dict]]:
    """
    Detect faces and get their embeddings using DeepFace.

    Returns:
        List of tuples: (face_image, face_embedding, face_region)
    """
    try:
        # Use DeepFace to extract faces and embeddings
        # represent() returns a list of face representations
        results = DeepFace.represent(
            img_path=str(image_path),
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False  # Don't fail if no face found
        )

        if not results:
            return []

        # Load the original image for cropping faces
        image = cv2.imread(str(image_path))
        if image is None:
            return []

        faces = []
        for result in results:
            embedding = result['embedding']
            face_region = result['facial_area']  # Dict with x, y, w, h

            # Extract face with padding
            x = face_region['x']
            y = face_region['y']
            w = face_region['w']
            h = face_region['h']

            # Add 20% padding
            padding = int(0.2 * max(w, h))
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)

            face_image = image[y1:y2, x1:x2]

            if face_image.size > 0:
                faces.append((face_image, embedding, face_region))

        return faces

    except Exception as e:
        # If DeepFace fails completely, return empty list
        return []


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def find_matching_person(embedding: List[float], persons_dataset: Dict) -> Tuple[Optional[str], float]:
    """
    Find a matching person in the dataset based on face embedding similarity.

    Returns:
        Tuple of (person_id, similarity) if match found, (None, best_similarity) otherwise
    """
    if not persons_dataset:
        return None, 0.0

    best_match = None
    best_similarity = 0.0
    expected_dim = len(embedding)

    for person_id, person_data in persons_dataset.items():
        if 'embedding' not in person_data or person_data['embedding'] is None:
            continue

        # Skip if embedding dimension doesn't match (e.g., from different model)
        if len(person_data['embedding']) != expected_dim:
            print(f"    ⚠ Skipping person {person_id}: incompatible embedding dimension ({len(person_data['embedding'])} vs {expected_dim})")
            continue

        similarity = cosine_similarity(embedding, person_data['embedding'])

        if similarity > best_similarity:
            best_similarity = similarity
            best_match = person_id

    if best_similarity >= SIMILARITY_THRESHOLD:
        return best_match, best_similarity

    return None, best_similarity


def detect_gender_deepface(face_image: np.ndarray) -> str:
    """
    Detect gender using DeepFace.
    Falls back to random if detection fails.
    """
    try:
        # Save face temporarily for DeepFace analysis
        temp_path = "/tmp/temp_face.png"
        cv2.imwrite(temp_path, face_image)

        # Analyze face attributes
        analysis = DeepFace.analyze(
            img_path=temp_path,
            actions=['gender'],
            enforce_detection=False,
            detector_backend=DETECTOR_BACKEND
        )

        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # Extract gender from analysis
        if isinstance(analysis, list):
            analysis = analysis[0]

        gender = analysis.get('dominant_gender', '').lower()
        if gender in ['man', 'male']:
            return 'male'
        elif gender in ['woman', 'female']:
            return 'female'
        else:
            return random.choice(['male', 'female'])

    except Exception as e:
        # Fallback to random if detection fails
        return random.choice(['male', 'female'])


def generate_person_metadata(gender: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate name and relationship for a person.
    Introduces randomness - 30% chance of unknown person (null values).

    Returns:
        (name, relationship) - can be (None, None) for unknown persons
    """
    # 30% chance of unknown person
    if random.random() < 0.3:
        return None, None

    # Generate name based on gender
    if gender == 'male':
        name = random.choice(MALE_NAMES)
        relationship = random.choice(MALE_RELATIONSHIPS)
    else:
        name = random.choice(FEMALE_NAMES)
        relationship = random.choice(FEMALE_RELATIONSHIPS)

    return name, relationship


def save_headshot(face_image: np.ndarray, person_id: str, headshots_dir: Path) -> bool:
    """Save face headshot image."""
    headshots_dir.mkdir(parents=True, exist_ok=True)
    output_path = headshots_dir / f"{person_id}.png"

    try:
        cv2.imwrite(str(output_path), face_image)
        return True
    except Exception as e:
        print(f"    ✗ Error saving headshot: {e}")
        return False


def process_images(images_dir: Path, persons_dir: Path):
    """
    Process all images to detect and recognize faces.
    """
    # Load persons dataset
    dataset_path = persons_dir / "dataset.json"
    persons_dataset = load_persons_dataset(dataset_path)

    # Create headshots directory
    headshots_dir = persons_dir / "headshots"
    headshots_dir.mkdir(parents=True, exist_ok=True)

    # Find all image files
    image_files = list(images_dir.glob("*/*.png"))

    if not image_files:
        print("No images found in the images directory")
        return

    print(f"\nFound {len(image_files)} images to process")
    print(f"Images directory: {images_dir.absolute()}")
    print(f"Persons directory: {persons_dir.absolute()}")
    print(f"Model: {MODEL_NAME}, Detector: {DETECTOR_BACKEND}")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD}\n")

    total_faces_detected = 0
    new_persons_added = 0
    existing_persons_matched = 0
    faces_skipped = 0

    for idx, image_path in enumerate(image_files, 1):
        youtube_video_id = image_path.parent.name
        image_id = image_path.stem

        print(f"[{idx}/{len(image_files)}] Processing {youtube_video_id}/{image_id}.png")

        try:
            # Detect faces and get embeddings
            faces = detect_and_encode_faces(image_path)

            if not faces:
                print(f"  No faces detected\n")
                continue

            print(f"  Detected {len(faces)} face(s)")
            total_faces_detected += len(faces)

            # Process each detected face
            for face_idx, (face_image, embedding, face_region) in enumerate(faces, 1):
                print(f"    Processing face {face_idx}/{len(faces)}...")

                # Try to match with existing persons
                person_id, similarity = find_matching_person(embedding, persons_dataset)

                if person_id is not None:
                    # Found existing person
                    person_name = persons_dataset[person_id].get('name', 'Unknown')

                    print(f"    ✓ Matched to existing person: {person_name} (ID: {person_id}, similarity: {similarity:.3f})")

                    # Add this media to associated media if not already there
                    if image_id not in persons_dataset[person_id]['associated-media']:
                        persons_dataset[person_id]['associated-media'].append(image_id)

                    existing_persons_matched += 1

                else:
                    # New person detected
                    new_person_id = str(uuid.uuid4())
                    gender = detect_gender_deepface(face_image)
                    name, relationship = generate_person_metadata(gender)

                    # Create new person entry
                    persons_dataset[new_person_id] = {
                        'id': new_person_id,
                        'name': name,
                        'relationship': relationship,
                        'associated-media': [image_id],
                        'embedding': embedding
                    }

                    # Save headshot
                    if save_headshot(face_image, new_person_id, headshots_dir):
                        name_str = name if name else "Unknown"
                        rel_str = relationship if relationship else "Unknown"
                        print(f"    ✓ New person added: {name_str} ({rel_str}) - ID: {new_person_id}, similarity: {similarity:.3f}")
                        new_persons_added += 1
                    else:
                        print(f"    ✗ Failed to save headshot for new person")

        except Exception as e:
            print(f"  ✗ Error processing image: {e}\n")
            continue

        print()

    # Save updated dataset
    save_persons_dataset(persons_dataset, dataset_path)

    # Print summary
    print("=" * 60)
    print(f"Processing complete!")
    print(f"Images processed: {len(image_files)}")
    print(f"Total faces detected: {total_faces_detected}")
    print(f"New persons added: {new_persons_added}")
    print(f"Existing persons matched: {existing_persons_matched}")
    print(f"Total persons in database: {len(persons_dataset)}")
    print(f"\nDataset saved to: {dataset_path}")
    print(f"Headshots saved to: {headshots_dir}/")


def main():
    """Main entry point."""
    print("Face Recognition System using DeepFace")
    print("=" * 60)

    # Set paths
    script_dir = Path(__file__).parent  # data/scripts/
    images_dir = script_dir.parent / "images"  # data/images/
    persons_dir = script_dir.parent / "persons"  # data/persons/

    if not images_dir.exists():
        print(f"Error: Images directory not found at {images_dir}")
        sys.exit(1)

    # Create persons directory if it doesn't exist
    persons_dir.mkdir(parents=True, exist_ok=True)

    # Process all images
    process_images(images_dir, persons_dir)


if __name__ == "__main__":
    main()
