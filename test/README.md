# Memorly Test Environment

This directory contains scripts and configuration for testing the Memorly system:
- **MongoDB setup and population** - Populate MongoDB with mock data
- **Vector DB population** - Process images and populate Milvus vector database

## Structure

The database uses a user UUID-based collection structure:
- **Database**: `memorly`
- **User UUID**: `mock-user`
- **Collections**:
  - `mock-user.media` - Media objects (images, videos, journals)
  - `mock-user.locations` - Location strings
  - `mock-user.persons` - Person objects with face embeddings

## Prerequisites

- Docker and Docker Compose
- Python 3.9+
- Required Python packages: `pymongo`

## Quick Start

### 1. Start MongoDB Server

```bash
cd test
docker-compose up -d
```

This will start a MongoDB instance on `localhost:27017` with:
- Database: `memorly`
- Persistent volumes for data and config
- Health checks enabled

### 2. Install Python Dependencies

```bash
pip install pymongo
```

### 3. Populate Database

```bash
python populate_mongodb.py
```

The script will:
- Connect to MongoDB at `mongodb://localhost:27017/`
- Load mock data from `../data/` directory:
  - `media.json` - Media objects
  - `locations.json` - Location list
  - `persons.json` - Person objects
- Create three collections under user `mock-user`:
  - `mock-user.media`
  - `mock-user.locations`
  - `mock-user.persons`
- Create indexes for efficient queries
- Display verification summary

### 4. Verify Population

The script automatically verifies the collections and shows:
- Document counts for each collection
- Sample documents from each collection
- Total processing summary

---

## Populate Vector Database (Milvus)

Test the complete image processing pipeline by populating the Milvus vector database with images from the data folder.

### Prerequisites

- All Memorly services running (`docker-compose up -d` in root directory)
- Gateway service healthy (check: `curl http://localhost:9000/health`)
- Milvus vector database accessible
- Python 3.9+
- Required packages: `requests`

### Quick Start

```bash
# Install dependencies
pip install requests

# Run the population script
python populate_vector_db.py
```

### What It Does

The `populate_vector_db.py` script:

1. **Checks gateway health** - Ensures all services are running
2. **Loads media metadata** - Reads `../data/media.json` for image metadata
3. **Processes each image** through the complete pipeline:
   - Uploads image to gateway service
   - Gateway extracts features (objects, content, tags) via Gemini
   - Gateway detects faces and creates/updates persons in MongoDB
   - Gateway generates CLIP embeddings
   - Gateway upserts to Milvus vector database
4. **Tracks statistics** - Shows success/failure rates and processing time

### Pipeline Flow for Each Image

```
Image File → Gateway Service
    ↓
Extract Features (Gemini)
    ↓
Extract Faces (ArcFace)
    ↓
Match/Create Persons (MongoDB)
    ↓
Generate Embedding (CLIP)
    ↓
Upsert to Vector DB (Milvus)
    ↓
Success!
```

### Example Output

```
============================================================
VECTOR DATABASE POPULATION SCRIPT
============================================================
Gateway URL: http://localhost:9000
User UUID:   mock-user
Data Dir:    /home/user/memorly/data
============================================================

Checking gateway health...
✓ Gateway is healthy
  ✓ extract-features
  ✓ face-extraction
  ✓ embed
  ✓ upsert
  ✓ video-segmentation
  ✓ mongodb

✓ Loaded 111 media entries

Found 111 images to process
============================================================

Process 111 images? [y/N]: y

Starting processing...

[1/111] e98093a9... (Chicago, IL)
  📁 images/jacyQPkeiZU/e98093a9-cbd4-4e58-a011-2288d8f6f186.png
  → Sending to gateway... ✓ (persons: +2/~0)

[2/111] 79f7db91... (Chicago, IL)
  📁 images/jacyQPkeiZU/79f7db91-8ca3-45ff-812e-0c4946c14ffe.png
  → Sending to gateway... ✓ (persons: +0/~2)

...

============================================================
PROCESSING STATISTICS
============================================================
Total images:           111
Successfully processed: 110 (99.1%)
Failed:                 1 (0.9%)
Skipped:                0 (0.0%)
Duration:               320.5 seconds
Average:                2.89 seconds per image
============================================================

✓ All images processed successfully!
```

### Configuration

Set the gateway URL if running on different host/port:

```bash
export GATEWAY_URL=http://localhost:9000
python populate_vector_db.py
```

### Verify Results

After population, check:

1. **MongoDB persons collection:**
```bash
docker exec -it memorly-mongodb mongosh memorly --eval "db['mock-user.persons'].countDocuments()"
```

2. **Milvus collection:**
Check via Milvus client or Attu UI

3. **Gateway logs:**
```bash
docker-compose logs -f gateway-service
```

### Troubleshooting

**Error: Gateway not accessible**
- Ensure services are running: `docker-compose ps`
- Check gateway health: `curl http://localhost:9000/health`

**Error: Request timeout**
- First few requests take longer (model downloads)
- Subsequent requests should be faster
- Increase timeout if processing large images

**Error: Image file not found**
- Verify images exist in `data/images/` directory
- Check `media.json` has correct file paths

## Environment Variables

You can customize the MongoDB connection and database:

```bash
export MONGO_URI="mongodb://localhost:27017/"
export DATABASE_NAME="memorly"
python populate_mongodb.py
```

## Database Schema

### mock-user.media

Documents from `media.json`:
```json
{
  "id": "e98093a9-cbd4-4e58-a011-2288d8f6f186",
  "timestamp": 1744237967,
  "fileFormat": "png",
  "mediaType": "image",
  "location": "Chicago, IL"
}
```

Indexes:
- `id` (unique)
- `timestamp`
- `mediaType`
- `location`

### mock-user.locations

Documents structure:
```json
{
  "location": "New York, NY"
}
```

Indexes:
- `location` (unique)

### mock-user.persons

Documents from `persons.json`:
```json
{
  "id": "f6dfb301-ae39-4e51-a28c-389567a92ce9",
  "name": null,
  "relationship": null,
  "associated-media": ["media-id-1", "media-id-2"],
  "embedding": [0.123, 0.456, ...]
}
```

Indexes:
- `id` (unique)
- `name`
- `associated-media`

## Managing MongoDB

### View Logs

```bash
docker-compose logs -f mongodb
```

### Stop MongoDB

```bash
docker-compose down
```

### Stop and Remove Data

```bash
docker-compose down -v
```

### Connect to MongoDB Shell

```bash
docker exec -it memorly-mongodb mongosh memorly
```

Then you can run MongoDB commands:
```javascript
// List all collections
show collections

// Count documents
db["mock-user.media"].countDocuments()

// Find sample documents
db["mock-user.media"].findOne()

// List all indexes
db["mock-user.media"].getIndexes()
```

## Troubleshooting

### Connection Refused

Make sure MongoDB is running:
```bash
docker-compose ps
```

If not running, start it:
```bash
docker-compose up -d
```

### Port Already in Use

If port 27017 is already in use, modify `docker-compose.yml`:
```yaml
ports:
  - "27018:27017"  # Use port 27018 instead
```

Then update the connection URI:
```bash
export MONGO_URI="mongodb://localhost:27018/"
python populate_mongodb.py
```

### Permission Errors

Ensure you have proper permissions to create Docker volumes:
```bash
sudo docker-compose up -d
```

## Data Files

Mock data is located in `../data/`:
- `media.json` - Dictionary of media objects keyed by ID
- `locations.json` - List of location strings
- `persons.json` - Dictionary of person objects keyed by ID

To regenerate the data, see the scripts in `../data/scripts/`.
