# Face Extraction Service

FastAPI service for detecting, extracting, and deduplicating faces from images using ArcFace embeddings and RetinaFace detection.

## Features

- **Face Detection**: Detect faces using RetinaFace (SOTA accuracy)
- **ArcFace Embeddings**: Generate 512-dimensional face embeddings using ArcFace model
- **Face Deduplication**: Remove duplicate faces using cosine similarity (threshold: 0.4)
- **Batch Processing**: Process multiple images in a single request
- **Headshot Extraction**: Return cropped, aligned face images (160x160 pixels)
- **Flexible Input**: Support for both URL and base64 encoded images

## Setup

### Option 1: Docker (Recommended)

1. **Build the Docker image**:
   ```bash
   docker build -t face-extraction-service .
   ```

2. **Run the container**:
   ```bash
   docker run -d \
     --name face-extraction \
     -p 5003:5003 \
     face-extraction-service
   ```

   **Note**: First run will download model weights (~300MB). This is a one-time operation.

3. **Check if it's running**:
   ```bash
   curl http://localhost:5003/health
   ```

### Option 2: Local Development

1. **Install system dependencies**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 libgl1-mesa-glx

   # macOS (OpenCV dependencies)
   brew install opencv
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the service**:
   ```bash
   python main.py
   ```

## API Endpoints

### 1. Extract Faces

**POST** `/extract-faces`

Extract unique faces from a list of images.

**Request Body**:
```json
{
  "images": [
    {"image_url": "https://example.com/photo1.jpg"},
    {"image_url": "https://example.com/photo2.jpg"},
    {"image_base64": "data:image/jpeg;base64,/9j/4AAQ..."}
  ],
  "similarity_threshold": 0.4,
  "min_face_size": 20
}
```

**Parameters**:
- `images` (array, required): List of images to process (max 100)
  - Each image can have either `image_url` or `image_base64`
- `similarity_threshold` (float, optional): Cosine similarity threshold for deduplication (0-1). Default: 0.4
  - Faces with similarity >= threshold are considered the same person
- `min_face_size` (int, optional): Minimum face size in pixels. Default: 20

**Response**:
```json
{
  "faces": [
    {
      "embedding": [0.123, -0.456, 0.789, ...],  // 512-dimensional vector
      "headshot_base64": "data:image/jpeg;base64,/9j/4AAQ...",
      "confidence": 0.99,
      "face_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    }
  ],
  "total_faces_detected": 15,
  "unique_faces": 8,
  "images_processed": 10,
  "request_id": "req_1729876543210",
  "processing_time_ms": 5420.3
}
```

### 2. Health Check

**GET** `/health`

Check if the service is running.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-10-25 21:00:00 UTC",
  "version": "1.0.0",
  "face_detection_model": "ArcFace + retinaface"
}
```

### 3. Readiness Check

**GET** `/ready`

Check if the service is ready to process requests.

**Response**:
```json
{
  "status": "ready",
  "timestamp": "2025-10-25 21:00:00 UTC",
  "version": "1.0.0",
  "face_detection_model": "ArcFace + retinaface"
}
```

## Interactive API Documentation

Once the service is running, visit:
- **Swagger UI**: http://localhost:5003/docs
- **ReDoc**: http://localhost:5003/redoc

## Example Usage

### Using cURL

```bash
# Extract faces from images
curl -X POST http://localhost:5003/extract-faces \
  -H "Content-Type: application/json" \
  -d '{
    "images": [
      {"image_url": "https://example.com/group_photo.jpg"},
      {"image_url": "https://example.com/another_photo.jpg"}
    ],
    "similarity_threshold": 0.4
  }'
```

### Using Python

```python
import requests

API_BASE_URL = "http://localhost:5003"

# Extract faces from URLs
response = requests.post(
    f"{API_BASE_URL}/extract-faces",
    json={
        "images": [
            {"image_url": "https://example.com/photo1.jpg"},
            {"image_url": "https://example.com/photo2.jpg"}
        ],
        "similarity_threshold": 0.4,
        "min_face_size": 20
    }
)

result = response.json()

print(f"Total faces detected: {result['total_faces_detected']}")
print(f"Unique faces: {result['unique_faces']}")

for i, face in enumerate(result['faces'], 1):
    print(f"\nFace {i}:")
    print(f"  ID: {face['face_id']}")
    print(f"  Confidence: {face['confidence']:.2%}")
    print(f"  Embedding dimension: {len(face['embedding'])}")
    print(f"  Headshot size: {len(face['headshot_base64'])} chars")
```

### Using with Local Files

```python
import requests
import base64

API_BASE_URL = "http://localhost:5003"

# Read local images
images = []
for image_path in ["photo1.jpg", "photo2.jpg", "photo3.jpg"]:
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
        images.append({
            "image_base64": f"data:image/jpeg;base64,{image_data}"
        })

# Send to API
response = requests.post(
    f"{API_BASE_URL}/extract-faces",
    json={
        "images": images,
        "similarity_threshold": 0.4
    }
)

faces = response.json()["faces"]

# Save headshots
for i, face in enumerate(faces, 1):
    # Extract base64 data
    headshot_data = face['headshot_base64'].split(',')[1]
    headshot_bytes = base64.b64decode(headshot_data)

    # Save to file
    with open(f"face_{i}_{face['face_id']}.jpg", "wb") as f:
        f.write(headshot_bytes)

    print(f"Saved face_{i}_{face['face_id']}.jpg")
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5003` | Port the service listens on |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `REQUEST_TIMEOUT` | `180` | Timeout for processing (seconds) |
| `MAX_IMAGES_PER_REQUEST` | `100` | Maximum images per batch request |
| `MIN_FACE_SIZE` | `20` | Minimum face size in pixels |
| `FACE_MODEL` | `ArcFace` | Face recognition model (ArcFace recommended) |
| `DETECTOR_BACKEND` | `retinaface` | Face detector (retinaface, mtcnn, opencv, ssd) |
| `ALLOWED_ORIGINS` | - | Comma-separated list of allowed CORS origins |

## How It Works

### 1. Face Detection
Uses **RetinaFace** detector to find faces in images:
- High accuracy face detection
- Detects face landmarks for alignment
- Provides bounding box coordinates
- Confidence scores for each detection

### 2. Face Embedding Generation
Uses **ArcFace** model to generate embeddings:
- 512-dimensional face embeddings
- State-of-the-art face recognition accuracy
- Embeddings are L2-normalized for cosine similarity
- Robust to variations in pose, lighting, expression

### 3. Face Deduplication
Removes duplicate faces using cosine similarity:
- Compares all face embeddings pairwise
- Similarity >= 0.4 → same person (configurable)
- Keeps first occurrence of each unique face
- Returns only unique faces with their embeddings

### 4. Output Format
For each unique face:
- **Embedding**: 512-dimensional normalized vector
- **Headshot**: 160x160 aligned face image (base64)
- **Confidence**: Detection confidence score
- **Face ID**: Unique identifier (UUID)

## Similarity Threshold Guide

The `similarity_threshold` controls how strictly faces are matched:

- **0.2-0.3**: Very strict - only nearly identical faces match
- **0.4**: **Recommended** - good balance, handles variations
- **0.5-0.6**: Lenient - may merge similar-looking different people
- **0.7+**: Very lenient - high false positive rate

## Response Schema

### Face

```json
{
  "embedding": [float],           // 512-dimensional normalized vector
  "headshot_base64": "string",    // Base64 encoded JPEG (160x160)
  "confidence": 0.99,             // Detection confidence (0-1)
  "face_id": "uuid"               // Unique identifier
}
```

## Error Handling

The service includes comprehensive error handling:

- **400 Bad Request**: Invalid input (missing image source, validation errors)
- **408 Request Timeout**: Processing took longer than timeout
- **500 Internal Server Error**: Unexpected errors during processing
- **503 Service Unavailable**: Service not ready

All errors return a structured error response:
```json
{
  "error": "Error message",
  "request_id": "req_1729876543210",
  "timestamp": "2025-10-25 21:00:00 UTC"
}
```

## Processing Pipeline

1. **Load Images**: Download from URLs or decode from base64
2. **Detect Faces**: Use RetinaFace to find all faces
3. **Filter**: Remove small faces and low-confidence detections
4. **Generate Embeddings**: Use ArcFace to create 512-d vectors
5. **Deduplicate**: Compare embeddings using cosine similarity
6. **Prepare Output**: Convert face images to base64, format response

## Performance Notes

- **Detection**: ~100-500ms per image (depends on resolution and face count)
- **Embedding**: ~50-100ms per face
- **Deduplication**: ~O(n²) where n = total faces detected
- **Memory**: ~300MB base + ~50MB per concurrent request
- **First run**: Downloads model weights (~300MB, one-time)

## Model Information

### ArcFace
- **Purpose**: Face recognition embeddings
- **Output**: 512-dimensional vectors
- **Accuracy**: State-of-the-art on LFW, CFP-FP, AgeDB
- **Size**: ~130MB

### RetinaFace
- **Purpose**: Face detection
- **Features**: Bounding boxes + 5 facial landmarks
- **Accuracy**: Best-in-class on WIDER FACE dataset
- **Size**: ~2MB

## Integration Example

Use with video segmentation service to extract faces from video scenes:

```python
import requests

# 1. Segment video into scenes
seg_response = requests.post(
    "http://localhost:5002/segment",
    json={"video_url": "https://example.com/video.mp4"}
)
scenes = seg_response.json()["scenes"]

# 2. Extract faces from scene frames
images = [{"image_base64": scene["frame_base64"]} for scene in scenes]

face_response = requests.post(
    "http://localhost:5003/extract-faces",
    json={"images": images, "similarity_threshold": 0.4}
)
unique_faces = face_response.json()["faces"]

print(f"Found {len(unique_faces)} unique people in the video")

# 3. Store face embeddings in database for future recognition
for face in unique_faces:
    # Store face embedding and headshot
    # Can be used later for face recognition across videos
    pass
```

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)
- WebP (.webp)

## Troubleshooting

### Out of memory
- Reduce `MAX_IMAGES_PER_REQUEST`
- Process fewer images per batch
- Increase Docker memory limits

### Slow processing
- Lower resolution images process faster
- Reduce number of faces per image
- Use GPU version (requires nvidia-docker)

### Model download fails
- Ensure internet connectivity on first run
- Models are cached in `~/.deepface/weights/`
- Manual download possible if needed

### Low-quality face detection
- Increase `min_face_size` to filter small faces
- Use higher resolution input images
- Ensure faces are clearly visible (not occluded)

## Notes

- All face images are resized to 160x160 pixels for consistency
- Embeddings are L2-normalized for cosine similarity computation
- Face detection confidence threshold is fixed at 0.5
- First API call may be slow due to model initialization
- Models are loaded into memory once at startup (after first use)
- Service automatically handles face alignment for better embeddings

## Citation

If you use this service in research, please cite:

```bibtex
@inproceedings{deng2019arcface,
  title={Arcface: Additive angular margin loss for deep face recognition},
  author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={CVPR},
  year={2019}
}

@inproceedings{deng2020retinaface,
  title={Retinaface: Single-shot multi-level face localisation in the wild},
  author={Deng, Jiankang and Guo, Jia and Ververas, Evangelos and Kotsia, Irene and Zafeiriou, Stefanos},
  booktitle={CVPR},
  year={2020}
}
```
