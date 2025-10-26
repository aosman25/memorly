# Gateway Service

Gateway service for the Memorly system that orchestrates the processing pipeline for images, videos, and text files. This service automates the entire workflow from media upload to vector database upsert, including feature extraction, face recognition, and embedding generation.

## Features

- **Multi-format Support**: Processes images, videos, and text
- **Automated Pipeline**: Orchestrates all microservices seamlessly
- **Face Recognition**: Detects faces and manages person associations in MongoDB
- **Vector Database Integration**: Automatically upserts embeddings to Milvus
- **RESTful API**: Easy-to-use endpoints for media processing

## Architecture

The gateway service coordinates the following microservices:

1. **extract-features-service**: Extracts objects, content, and tags from images
2. **face-extraction-service**: Detects and extracts face embeddings
3. **video-segmentation-service**: Segments videos into scenes with transcripts
4. **embed-service**: Generates CLIP embeddings for images, text, and videos
5. **upsert-service**: Manages Milvus vector database operations

## Processing Pipelines

### Image Pipeline

```
Image Upload → Extract Features → Extract Faces → Match/Create Persons → Generate Embedding → Upsert to Vector DB
```

1. Extract features (objects, content, tags) using Gemini
2. Extract faces and generate face embeddings using ArcFace
3. Match faces against existing persons in MongoDB (cosine similarity ≥ 0.4)
   - If match found: Add media ID to person's `associated-media` field
   - If no match: Create new person with face embedding
4. Generate image embedding using CLIP
5. Upsert all data to Milvus vector database

### Video Pipeline

```
Video Upload → Segment Video → Extract Features → Extract Faces → Match/Create Persons → Generate Fused Embedding → Upsert to Vector DB
```

1. Segment video into scenes using ffmpeg
2. Extract features from representative frames
3. Extract faces from video frames
4. Match/create persons (same as image pipeline)
5. Generate fused embedding (60% visual + 40% text from transcripts)
6. Upsert to vector database with video timestamps

### Text Pipeline

```
Text Upload → Generate Embedding → Upsert to Vector DB
```

1. Generate text embedding using CLIP
2. Upsert to vector database

## API Endpoints

### Health Check

```http
GET /health
```

Returns the health status of the gateway and all connected services.

**Response:**
```json
{
  "status": "healthy",
  "services": {
    "extract-features": true,
    "face-extraction": true,
    "embed": true,
    "upsert": true,
    "video-segmentation": true,
    "mongodb": true
  }
}
```

### Process Image

```http
POST /process/image
```

Process an image through the full pipeline.

**Form Data:**
- `file` (file): Image file to process
- `media_id` (string): Unique identifier for the media
- `user_id` (string): User UUID
- `timestamp` (integer): Unix timestamp when media was created
- `location` (string, optional): Location where media was captured

**Response:**
```json
{
  "success": true,
  "media_id": "e98093a9-cbd4-4e58-a011-2288d8f6f186",
  "message": "Image processed successfully",
  "persons_created": 2,
  "persons_updated": 1,
  "embedding_dimension": 512
}
```

### Process Video

```http
POST /process/video
```

Process a video through the full pipeline.

**Form Data:**
- `file` (file): Video file to process
- `media_id` (string): Unique identifier for the media
- `user_id` (string): User UUID
- `timestamp` (integer): Unix timestamp when media was created
- `location` (string, optional): Location where media was captured

**Response:**
```json
{
  "success": true,
  "media_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "message": "Video processed successfully (5 scenes)",
  "persons_created": 1,
  "persons_updated": 0,
  "embedding_dimension": 512
}
```

### Process Text

```http
POST /process/text
```

Process text through the pipeline.

**Form Data:**
- `text` (string): Text content to process
- `media_id` (string): Unique identifier for the media
- `user_id` (string): User UUID
- `timestamp` (integer): Unix timestamp when media was created
- `location` (string, optional): Location associated with the text

**Response:**
```json
{
  "success": true,
  "media_id": "text-123",
  "message": "Text processed successfully",
  "persons_created": null,
  "persons_updated": null,
  "embedding_dimension": 512
}
```

## Configuration

Configure the service using environment variables or a `.env` file.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `MONGO_URI` | MongoDB connection URI | `mongodb://localhost:27017/` |
| `MONGO_DB_NAME` | MongoDB database name | `memorly` |
| `EXTRACT_FEATURES_SERVICE_URL` | Feature extraction service URL | `http://localhost:8001` |
| `FACE_EXTRACTION_SERVICE_URL` | Face extraction service URL | `http://localhost:8002` |
| `EMBED_SERVICE_URL` | Embedding service URL | `http://localhost:8003` |
| `UPSERT_SERVICE_URL` | Upsert service URL | `http://localhost:8004` |
| `VIDEO_SEGMENTATION_SERVICE_URL` | Video segmentation service URL | `http://localhost:8005` |
| `FACE_SIMILARITY_THRESHOLD` | Cosine similarity threshold for face matching | `0.4` |
| `VIDEO_VISUAL_WEIGHT` | Weight for visual embeddings in video fusion | `0.6` |
| `VIDEO_TEXT_WEIGHT` | Weight for text embeddings in video fusion | `0.4` |
| `MAX_UPLOAD_SIZE` | Maximum file upload size in bytes | `100000000` (100MB) |
| `UPLOAD_DIR` | Temporary directory for uploaded files | `/tmp/gateway-uploads` |
| `LOG_LEVEL` | Logging level | `INFO` |

## Installation

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Copy the example environment file:
```bash
cp .env.example .env
```

3. Update `.env` with your configuration

4. Run the service:
```bash
python main.py
```

### Docker

1. Build the Docker image:
```bash
docker build -t gateway-service .
```

2. Run the container:
```bash
docker run -p 8000:8000 \
  -e MONGO_URI=mongodb://host.docker.internal:27017/ \
  -e EXTRACT_FEATURES_SERVICE_URL=http://host.docker.internal:8001 \
  -e FACE_EXTRACTION_SERVICE_URL=http://host.docker.internal:8002 \
  -e EMBED_SERVICE_URL=http://host.docker.internal:8003 \
  -e UPSERT_SERVICE_URL=http://host.docker.internal:8004 \
  -e VIDEO_SEGMENTATION_SERVICE_URL=http://host.docker.internal:8005 \
  gateway-service
```

## MongoDB Integration

The gateway service manages the `persons` collection in MongoDB for face recognition:

### Person Document Structure

```json
{
  "id": "f6dfb301-ae39-4e51-a28c-389567a92ce9",
  "name": null,
  "relationship": null,
  "associated-media": ["media-id-1", "media-id-2"],
  "embedding": [0.123, 0.456, ...]
}
```

### Face Matching Logic

1. For each face detected in an image/video:
   - Extract 512-dimensional ArcFace embedding
   - Compare with all existing persons using cosine similarity
   - If similarity ≥ 0.4: Add media ID to existing person's `associated-media`
   - If similarity < 0.4: Create new person with this face embedding

## Usage Examples

### Process an Image

```bash
curl -X POST http://localhost:8000/process/image \
  -F "file=@/path/to/image.jpg" \
  -F "media_id=img-123" \
  -F "user_id=mock-user" \
  -F "timestamp=1744237967" \
  -F "location=New York, NY"
```

### Process a Video

```bash
curl -X POST http://localhost:8000/process/video \
  -F "file=@/path/to/video.mp4" \
  -F "media_id=vid-456" \
  -F "user_id=mock-user" \
  -F "timestamp=1744237967" \
  -F "location=Los Angeles, CA"
```

### Process Text

```bash
curl -X POST http://localhost:8000/process/text \
  -F "text=This is a memory from my trip to Chicago" \
  -F "media_id=txt-789" \
  -F "user_id=mock-user" \
  -F "timestamp=1744237967" \
  -F "location=Chicago, IL"
```

### Check Health

```bash
curl http://localhost:8000/health
```

## Error Handling

The service includes comprehensive error handling:

- **400 Bad Request**: Invalid file type or missing required fields
- **500 Internal Server Error**: Processing errors (with detailed error messages)

All errors are logged with full stack traces for debugging.

## Logging

The service uses structured logging with the following levels:

- `DEBUG`: Detailed processing information
- `INFO`: High-level processing steps
- `WARNING`: Non-critical issues
- `ERROR`: Processing errors with stack traces

Configure logging level via the `LOG_LEVEL` environment variable.

## Dependencies

- **FastAPI**: Web framework
- **httpx**: Async HTTP client for service communication
- **pymongo**: MongoDB client
- **numpy**: Numerical operations for embeddings
- **uvicorn**: ASGI server

## Development

### Project Structure

```
gateway-service/
├── main.py                 # FastAPI application and endpoints
├── config.py              # Configuration management
├── models.py              # Pydantic models
├── pipelines.py           # Processing pipeline orchestration
├── service_clients.py     # Microservice client functions
├── mongodb_client.py      # MongoDB operations
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── .dockerignore         # Docker ignore patterns
├── .env.example          # Example environment configuration
└── README.md             # This file
```

### Adding New Features

To extend the gateway service:

1. **Add new pipeline logic** in [pipelines.py](pipelines.py)
2. **Add service client methods** in [service_clients.py](service_clients.py)
3. **Create new endpoints** in [main.py](main.py)
4. **Update models** in [models.py](models.py) if needed

## Troubleshooting

### Service Health Issues

Check the health endpoint to identify which services are unavailable:
```bash
curl http://localhost:8000/health
```

### MongoDB Connection Issues

Ensure MongoDB is running and accessible:
```bash
docker ps | grep mongodb
```

### File Upload Issues

Check the upload directory permissions and MAX_UPLOAD_SIZE setting.

### Processing Timeouts

Increase the timeout in [service_clients.py](service_clients.py:9) if processing large files.

## Performance Considerations

- **File Size Limits**: Default 100MB, adjust via `MAX_UPLOAD_SIZE`
- **Timeout**: 5 minutes for processing, 1 minute for connection
- **Cleanup**: Temporary files are automatically deleted after processing
- **Concurrent Requests**: FastAPI handles concurrent requests efficiently

## License

Part of the Memorly project.
