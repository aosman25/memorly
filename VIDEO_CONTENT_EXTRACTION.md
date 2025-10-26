# Video Content Extraction in Gateway Service

## Overview

The gateway service extracts content from videos through a multi-step pipeline that combines **visual analysis** and **audio transcription**.

## How Video Content is Extracted

### Pipeline Flow

```
Video File Upload
    ↓
1. VIDEO SEGMENTATION SERVICE
   - FFmpeg detects scene changes
   - Extracts keyframes for each scene
   - Whisper transcribes audio for each scene
   ↓
2. CONTENT EXTRACTION (Gateway assembles from transcripts)
   - Collects transcripts from ALL scenes
   - Joins them to create the video content
   ↓
3. FEATURE EXTRACTION (Visual features from first frame)
   - Gemini analyzes first frame
   - Extracts: objects, tags
   ↓
4. FACE EXTRACTION
   - Detects faces across all frames
   - Generates face embeddings
   ↓
5. EMBEDDING GENERATION
   - Fuses visual + text embeddings (60/40 ratio)
   ↓
6. UPSERT TO VECTOR DB
   - Stores: content, objects, tags, embedding
```

### Code Implementation

Located in: [`services/gateway-service/pipelines.py`](services/gateway-service/pipelines.py:215-230)

```python
# Step 1: Segment video into scenes
scenes = await self.service_clients.segment_video(media_path)

# Step 2: Extract content from scene transcripts
transcripts = []
for scene in scenes:
    transcript = scene.get("transcript", "")
    if transcript:
        transcripts.append(transcript)

# Join all transcripts as the video content
content = " ".join(transcripts) if transcripts else ""

# Extract visual features (objects, tags) from first frame
features = await self.service_clients.extract_features(media_path)
objects = features.get("objects", [])
tags = features.get("tags", [])
```

## Scene Structure

Each scene returned by the video segmentation service contains:

```json
{
  "start_time": 0.0,
  "end_time": 5.2,
  "frame": "base64_encoded_keyframe",
  "transcript": "This is what was said during this scene"
}
```

## Content Composition

The final content stored in the vector database is composed of:

1. **Content (text)**: All scene transcripts joined together
   - Source: Whisper audio transcription
   - Example: "Hello everyone. Welcome to my video. Today we're going to talk about..."

2. **Objects (list)**: Visual objects detected in the first frame
   - Source: Gemini vision analysis
   - Example: ["person", "laptop", "coffee mug", "window"]

3. **Tags (list)**: Categories and themes
   - Source: Gemini vision analysis
   - Example: ["tutorial", "technology", "home office"]

## Example Flow

### Input Video
- **Duration**: 30 seconds
- **Scenes**: 3 detected
- **Audio**: "Hi, I'm at the beach. The weather is beautiful today. Look at this sunset!"

### Processing

**Scene 1 (0-10s)**:
```json
{
  "start_time": 0.0,
  "end_time": 10.0,
  "transcript": "Hi, I'm at the beach."
}
```

**Scene 2 (10-20s)**:
```json
{
  "start_time": 10.0,
  "end_time": 20.0,
  "transcript": "The weather is beautiful today."
}
```

**Scene 3 (20-30s)**:
```json
{
  "start_time": 20.0,
  "end_time": 30.0,
  "transcript": "Look at this sunset!"
}
```

### Extracted Content

```json
{
  "content": "Hi, I'm at the beach. The weather is beautiful today. Look at this sunset!",
  "objects": ["person", "ocean", "sand", "sky", "sun"],
  "tags": ["beach", "sunset", "nature", "outdoors"],
  "start_timestamp_video": 0.0,
  "end_timestamp_video": 30.0
}
```

## Service Interactions

### 1. Video Segmentation Service
**Endpoint**: `POST /segment`

**Request**:
```json
{
  "video": "base64_encoded_video"
}
```

**Response**:
```json
{
  "scenes": [
    {
      "start_time": 0.0,
      "end_time": 5.2,
      "frame": "base64_frame",
      "transcript": "Scene dialogue..."
    },
    ...
  ]
}
```

### 2. Extract Features Service
**Endpoint**: `POST /extract`

**Request**:
```json
{
  "image": "base64_encoded_first_frame"
}
```

**Response**:
```json
{
  "objects": ["object1", "object2"],
  "content": "Gemini's description",
  "tags": ["tag1", "tag2"]
}
```

### 3. Embed Service
**Endpoint**: `POST /embed/video`

**Request**:
```json
{
  "frames": [
    {
      "frame": "base64_frame",
      "text": "transcript for this frame"
    }
  ],
  "visual_weight": 0.6,
  "text_weight": 0.4
}
```

**Response**:
```json
{
  "embedding": [0.123, 0.456, ...],  // 512-dimensional vector
  "dimension": 512
}
```

## Key Differences: Image vs Video Content

| Aspect | Image | Video |
|--------|-------|-------|
| **Content Source** | Gemini's visual description | Whisper audio transcription |
| **Objects Source** | Gemini analysis | Gemini analysis (first frame) |
| **Tags Source** | Gemini analysis | Gemini analysis (first frame) |
| **Embedding** | CLIP on image | Fused CLIP (60% visual + 40% text) |
| **Timestamps** | None | start_timestamp_video, end_timestamp_video |
| **Scenes** | N/A | Multiple scenes processed |

## Why This Approach?

### For Video Content (Transcripts):
✅ **Accurate**: Whisper provides precise audio-to-text conversion
✅ **Comprehensive**: Captures what was actually said in the video
✅ **Searchable**: Text content is easily searchable and semantically meaningful
✅ **Complete**: All scenes are included, not just one frame

### For Visual Features (Objects/Tags):
✅ **Efficient**: Analyzing every frame would be too slow/expensive
✅ **Representative**: First frame usually shows the video's context
✅ **Practical**: Objects and tags provide visual context without full analysis

## Configuration

Relevant settings in [`.env`](.env):

```env
# Video embedding fusion weights
VIDEO_VISUAL_WEIGHT=0.6  # Visual component
VIDEO_TEXT_WEIGHT=0.4    # Text (transcript) component

# Whisper API for transcription
WHISPER_API_KEY=${DEEPINFRA_API_KEY}
WHISPER_BASE_URL=https://api.deepinfra.com/v1/openai
```

## Testing Video Content Extraction

```bash
# Upload a video
curl -X POST http://localhost:9000/process/video \
  -F "file=@your-video.mp4" \
  -F "media_id=test-video-1" \
  -F "user_id=mock-user" \
  -F "timestamp=$(date +%s)" \
  -F "location=Los Angeles, CA"

# Response will show:
{
  "success": true,
  "media_id": "test-video-1",
  "message": "Video processed successfully (3 scenes)",
  "persons_created": 2,
  "persons_updated": 1,
  "embedding_dimension": 512
}
```

## Future Enhancements

Potential improvements to video content extraction:

1. **Speaker Diarization**: Identify who is speaking in the video
2. **Scene Descriptions**: Add Gemini descriptions for each scene (not just first)
3. **OCR**: Extract text visible in video frames
4. **Object Tracking**: Track object movements across scenes
5. **Emotion Detection**: Analyze emotional tone from audio
6. **Summary Generation**: Create concise summaries of longer videos

## Troubleshooting

### Empty Content
**Issue**: Video processed but content is empty
**Cause**: No audio or failed transcription
**Solution**: Check video has audio track, verify WHISPER_API_KEY

### Missing Objects/Tags
**Issue**: No visual features extracted
**Cause**: First frame extraction failed
**Solution**: Check GEMINI_API_KEY, verify video format

### Incomplete Transcripts
**Issue**: Only partial transcripts captured
**Cause**: Scene detection threshold too high
**Solution**: Adjust SCENE_THRESHOLD in video-segmentation-service

## Summary

**Video Content** in the Memorly system comes from:
- **Primary**: Audio transcription (Whisper) - The actual spoken words
- **Secondary**: Visual analysis (Gemini) - Objects and tags from first frame

This hybrid approach provides both **what was said** (content) and **what was seen** (objects/tags), creating a rich, searchable memory of the video.

---

**Updated**: 2025-10-26
**File**: [services/gateway-service/pipelines.py](services/gateway-service/pipelines.py)
