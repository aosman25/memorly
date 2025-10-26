"""
LLM Response Generation Service

This service generates conversational responses based on search results from the vector database.
It streams the response in chunks: first the retrieved sources metadata, then the LLM response.
"""

import os
import json
from typing import AsyncGenerator, Dict, Any, List
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
import httpx
import structlog
from pydantic import BaseModel, Field

# Configure structured logging
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

# Configuration
class Config:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
    SEARCH_SERVICE_URL: str = os.getenv("SEARCH_SERVICE_URL", "http://localhost:8007")
    PORT: int = int(os.getenv("PORT", "8000"))

    @classmethod
    def get_gemini_api_url(cls) -> str:
        """Get Gemini API URL with current model"""
        return f"https://generativelanguage.googleapis.com/v1beta/models/{cls.GEMINI_MODEL}:streamGenerateContent"


# Pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User's natural language query")
    user_id: str = Field(..., description="User UUID")
    limit: int = Field(default=5, ge=1, le=20, description="Maximum number of search results to retrieve")


# Initialize FastAPI app
app = FastAPI(
    title="LLM Response Generation Service",
    description="Generates conversational responses based on retrieved memories",
    version="1.0.0"
)

# HTTP client
http_client = httpx.AsyncClient(timeout=60.0)


# Prompt template
PROMPT_TEMPLATE = """You are an intelligent personal memory assistant that helps the user recall details, summarize moments, and answer questions from their stored multimodal memories.

Your task is to answer the user's query using the retrieved context below.
Each memory entry contains structured metadata such as modality (video, image, text), timestamps, people, locations, tags, and textual content.

Use only the information in the retrieved data to answer accurately.
If the context is insufficient, clearly say you don't have enough information.
Keep your response concise but informative.

---
USER QUERY:
{user_query}

---
RETRIEVED CONTEXT (from vector database):
{retrieved_data}

---
INSTRUCTIONS:
1. Read all retrieved entries carefully and synthesize a concise, factual, and human-readable answer to the query.
2. When possible, reference people, places, or timestamps to help the user remember.
3. If multiple memories seem related, merge them coherently.
4. If modality includes "video" or "image", infer actions or events described by the content.
5. Do not invent details that are not explicitly or implicitly supported by the retrieved data.
6. Be conversational and natural in your tone.

---
FINAL ANSWER:
"""


@app.on_event("startup")
async def startup_event():
    """Service startup event"""
    logger.info("Starting LLM Response Generation Service")
    if not Config.GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set - service may not work properly")


@app.on_event("shutdown")
async def shutdown_event():
    """Service shutdown event"""
    await http_client.aclose()
    logger.info("Service shutdown complete")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Check if Gemini API key is configured
    has_api_key = bool(Config.GEMINI_API_KEY)

    # Check search service health
    search_healthy = False
    try:
        response = await http_client.get(f"{Config.SEARCH_SERVICE_URL}/health", timeout=5.0)
        search_healthy = response.status_code == 200
    except Exception:
        pass

    all_healthy = has_api_key and search_healthy

    return {
        "status": "healthy" if all_healthy else "degraded",
        "gemini_api_configured": has_api_key,
        "search_service": search_healthy
    }


async def get_search_results(query: str, user_id: str, limit: int) -> Dict[str, Any]:
    """Retrieve search results from the search service"""
    try:
        response = await http_client.post(
            f"{Config.SEARCH_SERVICE_URL}/search",
            json={
                "query": query,
                "user_id": user_id,
                "limit": limit,
                "offset": 0
            }
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error("Search service error", status_code=e.response.status_code, error=e.response.text)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Search service error: {e.response.text}"
        )
    except Exception as e:
        logger.error("Failed to retrieve search results", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to retrieve search results: {str(e)}"
        )


def format_search_results_for_prompt(results: List[Dict[str, Any]]) -> str:
    """Format search results into a readable string for the LLM prompt"""
    if not results:
        return "No relevant memories found."

    formatted = []
    for idx, result in enumerate(results, 1):
        metadata = result.get("metadata", {})

        entry = f"Memory {idx}:"
        entry += f"\n  - ID: {result.get('id', 'unknown')}"
        entry += f"\n  - Type: {result.get('media_type', 'unknown')}"
        entry += f"\n  - Relevance Score: {result.get('score', 0):.2f}"

        if metadata.get("content"):
            entry += f"\n  - Content: {metadata['content'][:500]}..."  # Truncate long content

        if metadata.get("timestamp"):
            from datetime import datetime
            ts = datetime.fromtimestamp(metadata["timestamp"])
            entry += f"\n  - Date: {ts.strftime('%Y-%m-%d %H:%M:%S')}"

        if metadata.get("location"):
            entry += f"\n  - Location: {metadata['location']}"

        if metadata.get("person_ids"):
            entry += f"\n  - People: {len(metadata['person_ids'])} person(s)"

        if metadata.get("tags"):
            entry += f"\n  - Tags: {', '.join(metadata['tags'][:5])}"

        if metadata.get("objects"):
            entry += f"\n  - Objects: {', '.join(metadata['objects'][:5])}"

        formatted.append(entry)

    return "\n\n".join(formatted)


async def stream_gemini_response(prompt: str) -> AsyncGenerator[str, None]:
    """Stream response from Gemini API"""
    headers = {
        "x-goog-api-key": Config.GEMINI_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1024,
        },
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
    }

    try:
        api_url = Config.get_gemini_api_url()
        logger.debug("Calling Gemini API", url=api_url, model=Config.GEMINI_MODEL)

        async with http_client.stream(
            "POST",
            api_url,
            headers=headers,
            json=payload,
            timeout=60.0
        ) as response:
            response.raise_for_status()
            logger.debug("Gemini API response received", status=response.status_code)

            # Collect all response data
            response_data = b""
            async for chunk in response.aiter_bytes():
                response_data += chunk

            logger.debug("Received complete response", size=len(response_data))

            # Parse as JSON array
            try:
                response_list = json.loads(response_data.decode('utf-8'))
                logger.debug("Parsed response", type=type(response_list), length=len(response_list) if isinstance(response_list, list) else 0)

                # Response is an array - process each item
                for item in response_list:
                    # Check if content was blocked
                    if "promptFeedback" in item:
                        block_reason = item["promptFeedback"].get("blockReason")
                        if block_reason:
                            logger.warning("Gemini blocked content", reason=block_reason)
                            yield "I found relevant memories based on your query. Please refer to the retrieved sources above for details about the events, people, and locations in your memories."
                            return

                    # Extract text from candidates
                    if "candidates" in item:
                        for candidate in item["candidates"]:
                            if "content" in candidate:
                                parts = candidate["content"].get("parts", [])
                                for part in parts:
                                    if "text" in part:
                                        text = part["text"]
                                        logger.debug("Yielding text", length=len(text))
                                        yield text

            except json.JSONDecodeError as e:
                logger.error("Failed to parse Gemini response", error=str(e), response_preview=response_data[:500].decode('utf-8', errors='ignore'))
                raise

    except httpx.HTTPStatusError as e:
        logger.error("Gemini API error", status_code=e.response.status_code)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Gemini API error: {e.response.status_code}"
        )
    except Exception as e:
        logger.error("Failed to stream from Gemini", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate response: {str(e)}"
        )


async def generate_response_stream(request: QueryRequest) -> AsyncGenerator[bytes, None]:
    """
    Generate streaming response with two phases:
    1. First chunk: metadata with retrieved sources
    2. Subsequent chunks: LLM response text
    """
    import time
    start_time = time.time()

    logger.info("Processing query", query=request.query, user_id=request.user_id)

    try:
        # Step 1: Get search results
        logger.debug("Retrieving search results")
        search_response = await get_search_results(request.query, request.user_id, request.limit)

        results = search_response.get("results", [])
        query_info = search_response.get("query_info", {})

        # Step 2: Send metadata chunk with retrieved sources
        metadata_chunk = {
            "type": "metadata",
            "data": {
                "sources": results,
                "query_info": query_info,
                "total_results": len(results)
            }
        }
        yield f"data: {json.dumps(metadata_chunk)}\n\n".encode("utf-8")

        # Step 3: Format context for LLM
        formatted_context = format_search_results_for_prompt(results)
        prompt = PROMPT_TEMPLATE.format(
            user_query=request.query,
            retrieved_data=formatted_context
        )

        # Step 4: Stream LLM response
        logger.debug("Streaming LLM response")
        async for text_chunk in stream_gemini_response(prompt):
            response_chunk = {
                "type": "response",
                "data": text_chunk
            }
            yield f"data: {json.dumps(response_chunk)}\n\n".encode("utf-8")

        # Step 5: Send completion marker
        processing_time = (time.time() - start_time) * 1000
        done_chunk = {
            "type": "done",
            "data": {
                "processing_time_ms": processing_time
            }
        }
        yield f"data: {json.dumps(done_chunk)}\n\n".encode("utf-8")

        logger.info("Response generated", processing_time_ms=processing_time)

    except Exception as e:
        logger.error("Error generating response", error=str(e))
        error_chunk = {
            "type": "error",
            "data": {
                "error": str(e)
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n".encode("utf-8")


@app.post("/generate")
async def generate_response(request: QueryRequest):
    """
    Generate a conversational response based on the user's query.

    Returns a Server-Sent Events (SSE) stream with:
    1. Metadata chunk with retrieved sources
    2. Response chunks with LLM-generated text
    3. Done chunk with completion status
    """
    return StreamingResponse(
        generate_response_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=Config.PORT)
