# Memorly - AI-Powered Memory Management System

A comprehensive microservices-based system for managing and searching personal memories through images, videos, and text using AI-powered feature extraction, face recognition, and vector embeddings.

## Overview

Memorly automatically processes your media to extract meaningful information, recognize faces, and create searchable memories. The system uses state-of-the-art AI models for:

- **Visual Understanding**: Gemini 2.5 for object detection and scene description
- **Face Recognition**: ArcFace for face detection and identification
- **Multimodal Embeddings**: CLIP for unified image, text, and video embeddings
- **Video Processing**: FFmpeg for scene detection + Whisper for transcription
- **Vector Search**: Milvus for fast similarity search

## Features

- 🖼️ **Image Processing**: Extract objects, tags, faces, and generate embeddings
- 🎥 **Video Processing**: Scene segmentation, face detection, and fused visual-text embeddings
- 📝 **Text Processing**: Natural language understanding and embedding generation
- 👥 **Face Recognition**: Automatic person detection and association across media
- 🔍 **Vector Search**: Fast similarity search for finding related memories
- 🌍 **Location Tracking**: Associate memories with geographic locations
- 📊 **MongoDB Storage**: Structured storage for persons, media, and locations

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd memorly

# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
nano .env
```

### 2. Start Services

```bash
# Start all services
docker-compose up -d --build

# Check health
curl http://localhost:8000/health
```

### 3. Process Your First Memory

```bash
# Process an image
curl -X POST http://localhost:8000/process/image \
  -F "file=@photo.jpg" \
  -F "media_id=img-001" \
  -F "user_id=mock-user" \
  -F "timestamp=$(date +%s)" \
  -F "location=New York, NY"
```

See [QUICKSTART.md](QUICKSTART.md) for detailed setup instructions.

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Complete setup and usage guide
- **[API_KEYS.md](API_KEYS.md)** - API keys reference and setup
- **[GATEWAY_SETUP.md](GATEWAY_SETUP.md)** - Detailed gateway service documentation
- **[services/gateway-service/README.md](services/gateway-service/README.md)** - Gateway API reference
- **[test/README.md](test/README.md)** - MongoDB setup and testing
- **[.env.example](.env.example)** - Environment configuration reference

## API Endpoints

### Gateway Service (http://localhost:8000)

- `GET /health` - Health check for all services
- `POST /process/image` - Process an image
- `POST /process/video` - Process a video
- `POST /process/text` - Process text

Interactive API docs: http://localhost:8000/docs

## Architecture

### Microservices

1. **Gateway Service** (Port 8000) - Main entry point and orchestration
2. **Extract Features Service** (Port 8001) - Gemini-based feature extraction
3. **Face Extraction Service** (Port 8002) - ArcFace face recognition
4. **Embed Service** (Port 8003) - CLIP embedding generation
5. **Upsert Service** (Port 8004) - Milvus vector DB operations
6. **Video Segmentation Service** (Port 8005) - FFmpeg + Whisper

### Databases

- **MongoDB**: Stores persons, media metadata, and locations
- **Milvus**: Vector database for embedding-based similarity search

## Requirements

### System
- Docker & Docker Compose
- 8GB+ RAM recommended
- 10GB+ disk space

### API Keys
- **Gemini API**: For feature extraction
- **DeepInfra API**: For CLIP embeddings and Whisper transcription

## Configuration

All configuration is done via environment variables in `.env`:

```env
# API Keys (only 2 keys needed!)
GEMINI_API_KEY=your-gemini-key
DEEPINFRA_API_KEY=your-deepinfra-key

# Milvus (separate network)
MILVUS_HOST=host.docker.internal
MILVUS_PORT=19530

# Face Recognition
FACE_SIMILARITY_THRESHOLD=0.4
```

**Note**: `OPENAI_API_KEY` is automatically set to use DeepInfra's OpenAI-compatible API.

See [.env.example](.env.example) for all configuration options.

## Support

For issues and questions:
1. Check the [QUICKSTART.md](QUICKSTART.md)
2. Review service logs: `docker-compose logs -f`
3. Verify API keys in `.env`
4. Check service health: `curl http://localhost:8000/health`