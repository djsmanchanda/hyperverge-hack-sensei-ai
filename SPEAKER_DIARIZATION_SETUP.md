# Speaker Diarization Setup Guide

## Overview
This implementation adds speaker diarization capabilities to detect multiple speakers in audio recordings. It uses the state-of-the-art pyannote.audio library for accurate speaker segmentation.

## Installation Steps

### 1. Install Required Packages
```bash
pip install pyannote.audio torch torchaudio soundfile
```

### 2. Set Up HuggingFace Token
Speaker diarization requires access to pre-trained models from HuggingFace:

1. Create account at https://huggingface.co/
2. Get your access token from https://huggingface.co/settings/tokens
3. Set environment variable:
   ```bash
   # Windows
   set HUGGINGFACE_TOKEN=your_token_here
   
   # Linux/Mac
   export HUGGINGFACE_TOKEN=your_token_here
   ```
4. Accept the license for the diarization model at:
   https://huggingface.co/pyannote/speaker-diarization-3.1

### 3. Environment Configuration
Add to your `.env` file:
```
HUGGINGFACE_TOKEN=your_actual_token_here
```

## API Endpoints

### 1. Speaker Detection Only
```
POST /ai/audio/speaker-detection
```

**Parameters:**
- `audio_uuid`: UUID of uploaded audio file
- `max_duration`: Maximum seconds to process (default: 120)

**Response:**
```json
{
  "success": true,
  "audio_uuid": "uuid-here",
  "speaker_detection": {
    "num_speakers": 2,
    "speakers": ["SPEAKER_00", "SPEAKER_01"],
    "speaker_segments": [
      {
        "speaker": "SPEAKER_00",
        "start": 0.0,
        "end": 15.3,
        "duration": 15.3
      }
    ],
    "speaker_stats": {
      "SPEAKER_00": {
        "total_speaking_time": 25.4,
        "num_segments": 3,
        "percentage_of_total": 60.2
      }
    },
    "is_multi_speaker": true,
    "confidence": "high",
    "method": "pyannote_diarization"
  }
}
```

### 2. Enhanced Audio Evaluation with Speakers
```
POST /ai/audio/enhanced-evaluation-with-speakers
```

**Parameters:**
- `audio_uuid`: UUID of uploaded audio file
- `question`: Optional context question
- `context`: Optional additional context
- `include_speaker_analysis`: Boolean (default: true)

**Response:** Combined transcription, speech analysis, and speaker detection

## Features

### Speaker Detection Capabilities
- **Multi-speaker identification**: Detects and labels individual speakers
- **Speaker segmentation**: Provides timestamps for each speaker's segments
- **Speaking time analysis**: Calculates speaking duration and percentages
- **Confidence scoring**: Indicates reliability of detection

### Fallback Modes
1. **High-quality mode**: Uses pyannote.audio for accurate diarization
2. **Estimation mode**: Simple audio analysis when pyannote unavailable
3. **Basic fallback**: Assumes single speaker when no analysis possible

### Performance Optimizations
- **Duration limiting**: Process only first N seconds for efficiency
- **Mono conversion**: Automatic conversion to mono audio
- **Sample rate optimization**: Uses 16kHz for faster processing

## Integration Examples

### Frontend Integration
```typescript
// Check if multiple speakers detected
const checkSpeakers = async (audioUuid: string) => {
  const response = await fetch('/ai/audio/speaker-detection', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ audio_uuid: audioUuid })
  });
  
  const result = await response.json();
  
  if (result.speaker_detection.is_multi_speaker) {
    showMultiSpeakerWarning(result.speaker_detection.num_speakers);
  }
};
```

### Backend Integration
```python
from api.utils.audio import enhanced_audio_service

# Get comprehensive analysis
result = enhanced_audio_service.transcribe_with_speaker_analysis(audio_data)

if result['speaker_analysis']['is_multi_speaker']:
    # Handle multi-speaker scenario
    num_speakers = result['speaker_analysis']['num_speakers']
    print(f"Detected {num_speakers} speakers")
```

## Troubleshooting

### Common Issues

1. **"Pipeline not available"**
   - Install pyannote.audio: `pip install pyannote.audio`
   - Set HUGGINGFACE_TOKEN environment variable
   - Accept model license on HuggingFace

2. **"Import torch failed"**
   - Install PyTorch: `pip install torch torchaudio`
   - For GPU support: Follow PyTorch installation guide

3. **"Model loading failed"**
   - Check internet connection
   - Verify HuggingFace token is valid
   - Ensure model license is accepted

4. **"Fallback mode used"**
   - This is normal when pyannote is unavailable
   - Results will be less accurate but still functional

### Performance Tips

1. **For faster processing:**
   - Limit max_duration to 60-120 seconds
   - Use mono audio files
   - Ensure adequate RAM (4GB+ recommended)

2. **For better accuracy:**
   - Use high-quality audio (clear, minimal background noise)
   - Ensure speakers have distinct voices
   - Avoid overlapping speech when possible

## Testing

Run the test script to verify installation:
```bash
python test_speaker_detection.py
```

This will test:
- Package availability
- Basic speaker detection
- Enhanced transcription
- Fallback mechanisms

## Architecture Notes

The implementation includes three service classes:

1. **AudioTranscriptionService**: Base transcription with OpenAI Whisper
2. **SpeakerDiarizationService**: Dedicated speaker detection
3. **EnhancedAudioTranscriptionService**: Combined functionality

This modular design allows:
- Graceful degradation when components unavailable
- Easy testing and maintenance
- Flexible integration options
