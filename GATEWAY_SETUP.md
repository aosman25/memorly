# Gateway Service Setup Guide

This guide explains how to use the newly created Gateway Service for the Memorly system.

## Overview

The Gateway Service is the main entry point for processing media (images, videos, text) in the Memorly system. It orchestrates all microservices to:

1. Extract features and metadata
2. Detect and recognize faces
3. Generate embeddings
4. Store data in MongoDB and Milvus

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Gateway Service                           │
│                     (Port 8000)                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
        ┌───────▼──────┐ ┌───▼────┐ ┌─────▼─────┐
        │   MongoDB    │ │ Milvus │ │ Services  │
        │  (Persons)   │ │ (Vec)  │ │           │
        └──────────────┘ └────────┘ └─────┬─────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
            ┌───────▼────────┐   ┌────────▼────────┐   ┌────────▼────────┐
            │  Extract       │   │  Face           │   │  Video          │
            │  Features      │   │  Extraction     │   │  Segmentation   │
            │  (Port 8001)   │   │  (Port 8002)    │   │  (Port 8005)    │
            └────────────────┘   └─────────────────┘   └─────────────────┘
                    │                      │                      │
            ┌───────▼────────┐   ┌────────▼────────┐
            │  Embed         │   │  Upsert         │
            │  Service       │   │  Service        │
            │  (Port 8003)   │   │  (Port 8004)    │
            └────────────────┘   └─────────────────┘
```

## Processing Flows

### Image Processing Flow

```
User uploads image
    ↓
Gateway Service receives file
    ↓
1. Extract Features Service
   → Gemini extracts: objects, content, tags
    ↓
2. Face Extraction Service
   → RetinaFace detects faces
   → ArcFace generates face embeddings (512-d)
    ↓
3. MongoDB Person Matching
   → Compare with existing persons (cosine similarity ≥ 0.4)
   → Update existing person OR create new person
    ↓
4. Embed Service
   → CLIP generates image embedding (512-d)
    ↓
5. Upsert Service
   → Store embedding + metadata in Milvus
    ↓
Return success response
```

### Video Processing Flow

```
User uploads video
    ↓
Gateway Service receives file
    ↓
1. Video Segmentation Service
   → FFmpeg detects scene changes
   → Extract keyframes
   → Whisper transcribes audio
    ↓
2. Extract Features Service
   → Process representative frames
    ↓
3. Face Extraction Service
   → Extract faces from video frames
    ↓
4. MongoDB Person Matching
   → Match/create persons (same as image)
    ↓
5. Embed Service
   → Generate fused embedding
   → 60% visual (CLIP on frames) + 40% text (CLIP on transcripts)
    ↓
6. Upsert Service
   → Store with video timestamps
    ↓
Return success response
```

### Text Processing Flow

```
User submits text
    ↓
Gateway Service receives text
    ↓
1. Embed Service
   → CLIP generates text embedding (512-d)
    ↓
2. Upsert Service
   → Store embedding + text in Milvus
    ↓
Return success response
```

## Quick Start

### Prerequisites

1. All microservices built and ready:
   - extract-features-service
   - face-extraction-service
   - embed-service
   - upsert-service
   - video-segmentation-service

2. MongoDB running (for persons collection)
3. Milvus running (for vector database)

### Option 1: Docker Compose (Recommended)

1. Set up environment variables:
```bash
export GEMINI_API_KEY="your-gemini-api-key"
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_BASE_URL="your-openai-base-url"
export DEEPINFRA_API_KEY="your-deepinfra-api-key"
```

2. Start all services:
```bash
docker-compose up -d
```

3. Check health:
```bash
curl http://localhost:8000/health
```

### Option 2: Local Development

1. Start MongoDB:
```bash
cd test
docker-compose up -d
```

2. Start each microservice individually (in separate terminals):
```bash
# Extract Features Service
cd services/extract-features-service
python main.py

# Face Extraction Service
cd services/face-extraction-service
python main.py

# Embed Service
cd services/embed-service
python main.py

# Upsert Service
cd services/upsert-service
python main.py

# Video Segmentation Service
cd services/video-segmentation-service
python main.py

# Gateway Service
cd services/gateway-service
python main.py
```

3. Check health:
```bash
curl http://localhost:8000/health
```

## API Usage

### 1. Process an Image

```bash
curl -X POST http://localhost:8000/process/image \
  -F "file=@/path/to/photo.jpg" \
  -F "media_id=img-$(date +%s)" \
  -F "user_id=mock-user" \
  -F "timestamp=$(date +%s)" \
  -F "location=New York, NY"
```

Response:
```json
{
  "success": true,
  "media_id": "img-1744237967",
  "message": "Image processed successfully",
  "persons_created": 2,
  "persons_updated": 1,
  "embedding_dimension": 512
}
```

### 2. Process a Video

```bash
curl -X POST http://localhost:8000/process/video \
  -F "file=@/path/to/video.mp4" \
  -F "media_id=vid-$(date +%s)" \
  -F "user_id=mock-user" \
  -F "timestamp=$(date +%s)" \
  -F "location=Los Angeles, CA"
```

### 3. Process Text

```bash
curl -X POST http://localhost:8000/process/text \
  -F "text=Had an amazing dinner at the Italian restaurant downtown" \
  -F "media_id=txt-$(date +%s)" \
  -F "user_id=mock-user" \
  -F "timestamp=$(date +%s)" \
  -F "location=Chicago, IL"
```

### 4. Health Check

```bash
curl http://localhost:8000/health | jq
```

Response shows status of all services:
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

## Testing

Use the provided test script:

```bash
cd services/gateway-service

# Test health and text processing
python test_gateway.py

# Test with an image
python test_gateway.py /path/to/test-image.jpg
```

## MongoDB Person Management

The gateway automatically manages the persons collection:

### View Persons

```bash
docker exec -it memorly-mongodb mongosh memorly --eval "db['mock-user.persons'].find().pretty()"
```

### Check Person Associations

```bash
docker exec -it memorly-mongodb mongosh memorly --eval "
  db['mock-user.persons'].find(
    {'associated-media': {$exists: true}},
    {id: 1, name: 1, 'associated-media': 1}
  ).pretty()
"
```

### Count Persons

```bash
docker exec -it memorly-mongodb mongosh memorly --eval "db['mock-user.persons'].countDocuments()"
```

## Configuration

Gateway service configuration in `services/gateway-service/.env`:

```env
# Service URLs (adjust for your environment)
EXTRACT_FEATURES_SERVICE_URL=http://localhost:8001
FACE_EXTRACTION_SERVICE_URL=http://localhost:8002
EMBED_SERVICE_URL=http://localhost:8003
UPSERT_SERVICE_URL=http://localhost:8004
VIDEO_SEGMENTATION_SERVICE_URL=http://localhost:8005

# MongoDB
MONGO_URI=mongodb://localhost:27017/
MONGO_DB_NAME=memorly

# Face matching threshold (0.0 to 1.0)
FACE_SIMILARITY_THRESHOLD=0.4

# Video embedding weights
VIDEO_VISUAL_WEIGHT=0.6
VIDEO_TEXT_WEIGHT=0.4

# Upload limits
MAX_UPLOAD_SIZE=100000000  # 100MB
```

## Troubleshooting

### All Services Show Unhealthy

Check each service individually:
```bash
curl http://localhost:8001/health  # extract-features
curl http://localhost:8002/health  # face-extraction
curl http://localhost:8003/health  # embed
curl http://localhost:8004/health  # upsert
curl http://localhost:8005/health  # video-segmentation
```

### MongoDB Connection Errors

Ensure MongoDB is running:
```bash
docker ps | grep mongodb
```

### Face Recognition Not Working

Check the similarity threshold. Lower it for looser matching:
```env
FACE_SIMILARITY_THRESHOLD=0.3
```

### Processing Timeouts

For large files, increase timeout in `service_clients.py`:
```python
self.timeout = httpx.Timeout(600.0, connect=60.0)  # 10 min
```

## Performance Notes

- **Image processing**: ~5-15 seconds (depending on face count)
- **Video processing**: ~30-120 seconds (depending on length)
- **Text processing**: ~1-2 seconds

Processing times depend on:
- File size
- Number of faces detected
- Video length and scene count
- API response times (Gemini, Whisper)

## Next Steps

1. **Batch Processing**: Create a batch endpoint for multiple files
2. **Webhooks**: Add webhook support for async processing notifications
3. **Search API**: Create search endpoints for querying the vector database
4. **Analytics**: Add processing statistics and monitoring
5. **Face Labeling**: Add endpoints to update person names/relationships

## File Locations

- **Gateway Service**: `services/gateway-service/`
- **Main Compose File**: `docker-compose.yml`
- **Test Compose**: `test/docker-compose.yml` (MongoDB only)
- **API Documentation**: `services/gateway-service/README.md`

## Support

For issues or questions:
1. Check service health endpoints
2. Review logs: `docker-compose logs -f gateway-service`
3. Verify all environment variables are set
4. Ensure all dependent services are running

---

**Gateway Service v1.0.0** - Part of the Memorly Memory System
