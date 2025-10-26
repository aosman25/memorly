# Query Processing Service

A microservice that processes user search queries to extract structured information for RAG (Retrieval-Augmented Generation) search.

## Overview

This service analyzes natural language search queries and extracts:
- **Objects**: Physical items or entities mentioned (e.g., "photos", "laptop", "coffee")
- **Tags**: Categories and descriptors (e.g., "vacation", "outdoor", "summer")
- **Content**: Cleaned and normalized query text
- **People IDs**: Matches person names to IDs from MongoDB persons collection
- **Location IDs**: Matches location names to IDs from MongoDB locations collection

## Architecture

- **Framework**: FastAPI (Python 3.11)
- **AI Model**: Google Gemini 2.5 Flash
- **Database**: MongoDB for person and location lookups
- **Port**: 8006 (external), 8000 (internal)

## API Endpoints

### POST /process-query

Process a user search query.

**Request:**
```json
{
  "query": "Show me photos with John from Paris",
  "user_id": "user-uuid"
}
```

**Response:**
```json
{
  "objects": ["photos"],
  "tags": ["vacation", "travel"],
  "content": "photos with John from Paris",
  "people_ids": ["person-john-123"],
  "location_ids": ["location-paris-001"],
  "request_id": "req_1234567890",
  "processing_time_ms": 2500.0
}
```

### GET /health

Health check endpoint.

### GET /ready

Readiness check endpoint.

## Environment Variables

Required:
- `GEMINI_API_KEY`: Google Gemini API key
- `MONGO_URI`: MongoDB connection URI (default: mongodb://localhost:27017)
- `MONGO_DB_NAME`: MongoDB database name (default: memorly)

Optional:
- `LOG_LEVEL`: Logging level (default: INFO)
- `REQUEST_TIMEOUT`: Request timeout in seconds (default: 30)
- `GEMINI_MODEL`: Gemini model to use (default: gemini-2.5-flash)
- `PORT`: Service port (default: 8000)

## MongoDB Collections

The service reads from the following MongoDB collections:

### Persons Collection (`{user_id}.persons`)
```json
{
  "id": "person-uuid",
  "name": "John",
  "relationship": "friend",
  "associated-media": ["media-id-1"],
  "embedding": [0.1, 0.2, ...]
}
```

### Locations Collection (`{user_id}.locations`)
```json
{
  "id": "location-uuid",
  "name": "Paris",
  "location": "Paris, France",
  "country": "France"
}
```

## How It Works

1. **Query Extraction**: Uses Gemini AI to analyze the query and extract structured information
2. **Person Matching**: Searches MongoDB persons collection for names mentioned in the query
3. **Location Matching**: Searches MongoDB locations collection for locations mentioned in the query
4. **Response**: Returns structured data ready for embedding and filtering in RAG search

## Example Queries

```python
# Query with person and location
"Show me photos with John from our trip to Paris last summer"
# Returns: people_ids=["john-id"], location_ids=["paris-id"]

# Query with multiple people
"Find videos of my birthday party with Sarah and Mike"
# Returns: people_ids=["sarah-id", "mike-id"]

# Query with location
"Pictures from the beach at sunset"
# Returns: location_ids=["beach-id"]

# Query with objects
"Show me all images with my laptop and coffee"
# Returns: objects=["images", "laptop", "coffee"]
```

## Testing

Run the test suite:

```bash
# Basic functionality test
python3 test/test_query_processing.py

# Test with MongoDB matching
python3 test/test_query_matching.py
```

## Integration with RAG Search

The structured output from this service can be used for:

1. **Embedding Generation**: Create query embedding from the normalized content
2. **Metadata Filtering**: Filter search results by:
   - People IDs (exact match)
   - Location IDs (exact match)
   - Objects (contains)
   - Tags (contains)
3. **Semantic Search**: Use embedding for similarity search in vector database

## Development

Build and run:

```bash
# Build Docker image
docker-compose -f docker-compose.dev.yml build query-processing-service

# Start service
docker-compose -f docker-compose.dev.yml up -d query-processing-service

# View logs
docker logs query-processing-service

# Stop service
docker-compose -f docker-compose.dev.yml stop query-processing-service
```

## Performance

- Average processing time: 1.5-2.5 seconds per query
- Retry logic: 3 attempts with exponential backoff
- Timeout: 30 seconds (configurable)
- Concurrent request support: Yes

## Error Handling

- Invalid input: 400 Bad Request
- Timeout: 408 Request Timeout
- Service unavailable: 503 Service Unavailable
- Internal errors: 500 Internal Server Error

All errors include:
- Error message
- Request ID for tracking
- Timestamp
