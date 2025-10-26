# Memorly System - Quick Start Guide

Complete guide to set up and run the Memorly memory system with all microservices.

## Prerequisites

- Docker and Docker Compose installed
- Milvus vector database running (on separate network)
- API Keys:
  - Gemini API key (for image feature extraction)
  - DeepInfra API key (for CLIP embeddings and Whisper transcription)

## Step 1: Environment Setup

1. **Copy the environment template:**
```bash
cp .env.example .env
```

2. **Edit the `.env` file and add your API keys:**
```bash
nano .env
```

Required variables:
```env
# API Keys (only 2 keys needed!)
GEMINI_API_KEY=your-gemini-api-key-here
DEEPINFRA_API_KEY=your-deepinfra-api-key-here

# Note: OPENAI_API_KEY and OPENAI_BASE_URL are automatically configured
# to use DeepInfra's OpenAI-compatible endpoint

# Milvus Configuration (update to match your Milvus instance)
MILVUS_HOST=host.docker.internal
MILVUS_PORT=19530

# MongoDB Configuration (defaults are fine for docker-compose)
MONGO_URI=mongodb://mongodb:27017/
MONGO_DB_NAME=memorly

# Gateway Configuration (optional - defaults shown)
FACE_SIMILARITY_THRESHOLD=0.4
VIDEO_VISUAL_WEIGHT=0.6
VIDEO_TEXT_WEIGHT=0.4
MAX_UPLOAD_SIZE=100000000
LOG_LEVEL=INFO
```

## Step 2: Start Milvus (if not already running)

If you haven't set up Milvus yet:

```bash
# Download Milvus standalone docker-compose
wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O milvus-docker-compose.yml

# Start Milvus
docker-compose -f milvus-docker-compose.yml up -d
```

Verify Milvus is running:
```bash
docker ps | grep milvus
```

## Step 3: Start Memorly Services

Build and start all services:

```bash
docker-compose up -d --build
```

This will start:
- MongoDB (port 27017)
- Extract Features Service (port 8001)
- Face Extraction Service (port 8002)
- Embed Service (port 8003)
- Upsert Service (port 8004)
- Video Segmentation Service (port 8005)
- Gateway Service (port 8000)

## Step 4: Verify Services

Check all services are healthy:

```bash
curl http://localhost:8000/health | jq
```

Expected output:
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

Check individual service logs:
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f gateway-service
```

## Step 5: Populate MongoDB with Mock Data

```bash
cd test
python3 populate_mongodb.py
```

Verify data:
```bash
docker exec -it memorly-mongodb mongosh memorly --eval "show collections"
```

## Step 6: Test the System

### Test Image Processing

```bash
curl -X POST http://localhost:8000/process/image \
  -F "file=@/path/to/your/image.jpg" \
  -F "media_id=test-img-$(date +%s)" \
  -F "user_id=mock-user" \
  -F "timestamp=$(date +%s)" \
  -F "location=New York, NY"
```

### Test Text Processing

```bash
curl -X POST http://localhost:8000/process/text \
  -F "text=Had a wonderful dinner at the Italian restaurant" \
  -F "media_id=test-txt-$(date +%s)" \
  -F "user_id=mock-user" \
  -F "timestamp=$(date +%s)" \
  -F "location=Chicago, IL"
```

### Test Video Processing

```bash
curl -X POST http://localhost:8000/process/video \
  -F "file=@/path/to/your/video.mp4" \
  -F "media_id=test-vid-$(date +%s)" \
  -F "user_id=mock-user" \
  -F "timestamp=$(date +%s)" \
  -F "location=Los Angeles, CA"
```

## Common Commands

### Start all services
```bash
docker-compose up -d
```

### Stop all services
```bash
docker-compose down
```

### Stop and remove all data
```bash
docker-compose down -v
```

### Rebuild a specific service
```bash
docker-compose up -d --build gateway-service
```

### View logs
```bash
# All services
docker-compose logs -f

# Gateway service
docker-compose logs -f gateway-service

# Last 100 lines
docker-compose logs --tail=100 gateway-service
```

### Restart a service
```bash
docker-compose restart gateway-service
```

### Check service status
```bash
docker-compose ps
```

## Troubleshooting

### Services show as unhealthy

1. Check individual service logs:
```bash
docker-compose logs gateway-service
```

2. Verify environment variables:
```bash
docker-compose config
```

3. Check API keys are set in `.env`

### MongoDB connection errors

Ensure MongoDB is running:
```bash
docker ps | grep mongodb
```

Restart MongoDB if needed:
```bash
docker-compose restart mongodb
```

### Milvus connection errors

Update `MILVUS_HOST` in `.env`:
- If Milvus is on host machine: `host.docker.internal`
- If Milvus is on same Docker network: `milvus`
- If Milvus is on remote host: `your.milvus.host`

### Port conflicts

If ports are already in use, modify `docker-compose.yml`:
```yaml
ports:
  - "8080:8000"  # Use port 8080 instead of 8000
```

### Out of memory

Increase Docker memory limits in Docker Desktop settings.

### Services not starting

Check Docker logs:
```bash
docker-compose logs
```

Rebuild from scratch:
```bash
docker-compose down -v
docker-compose up -d --build
```

## Service URLs

When services are running:

- **Gateway Service**: http://localhost:8000
  - Health: http://localhost:8000/health
  - Docs: http://localhost:8000/docs

- **Extract Features**: http://localhost:8001
- **Face Extraction**: http://localhost:8002
- **Embed Service**: http://localhost:8003
- **Upsert Service**: http://localhost:8004
- **Video Segmentation**: http://localhost:8005

## MongoDB Access

Connect to MongoDB shell:
```bash
docker exec -it memorly-mongodb mongosh memorly
```

View persons collection:
```javascript
db["mock-user.persons"].find().pretty()
```

Count documents:
```javascript
db["mock-user.media"].countDocuments()
db["mock-user.persons"].countDocuments()
db["mock-user.locations"].countDocuments()
```

## Next Steps

1. **Test with your own data**: Upload your images/videos
2. **Explore the API**: Visit http://localhost:8000/docs for interactive API documentation
3. **Monitor persons**: Watch how faces are detected and associated
4. **Query vectors**: Use the upsert service to search for similar memories
5. **Scale up**: Adjust resource limits in docker-compose.yml for production

## Architecture Overview

```
┌──────────────┐
│   Client     │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────┐
│   Gateway Service (8000)     │
│  - Orchestrates pipeline     │
│  - Manages face recognition  │
└──────┬───────────────────────┘
       │
       ├─────────────────────────────────────┐
       │                                     │
       ▼                                     ▼
┌─────────────┐                    ┌──────────────┐
│  MongoDB    │                    │  Microservices│
│  - Persons  │                    │  - Features   │
│  - Media    │                    │  - Faces      │
│  - Locations│                    │  - Embed      │
└─────────────┘                    │  - Video      │
                                   │  - Upsert     │
                                   └───────┬───────┘
                                           │
                                           ▼
                                   ┌──────────────┐
                                   │   Milvus     │
                                   │ Vector DB    │
                                   └──────────────┘
```

## Support

- Check service logs: `docker-compose logs -f [service-name]`
- Verify health: `curl http://localhost:8000/health`
- Review configuration: `docker-compose config`
- Consult individual service READMEs in `services/*/README.md`

## Additional Resources

- [Gateway Service Documentation](services/gateway-service/README.md)
- [MongoDB Setup Guide](test/README.md)
- [Detailed Setup Guide](GATEWAY_SETUP.md)
- [Environment Configuration](.env.example)
