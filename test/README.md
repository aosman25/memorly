# MongoDB Test Environment

This directory contains scripts and configuration for setting up and populating a local MongoDB instance with mock data for testing.

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
