#!/usr/bin/env python3
"""
Test script for speaker diarization functionality
"""

import os
import sys
import tempfile
import numpy as np
import soundfile as sf

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from api.utils.audio import SpeakerDiarizationService, enhanced_audio_service


def create_test_audio_file():
    """Create a simple test audio file for testing"""
    sample_rate = 16000
    duration = 10  # 10 seconds
    
    # Generate simple sine wave audio
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a simple pattern that might simulate different speakers
    # First 5 seconds: higher frequency (speaker 1)
    # Last 5 seconds: lower frequency (speaker 2)
    frequency_1 = 440  # A4 note
    frequency_2 = 220  # A3 note (octave lower)
    
    audio_data = np.zeros_like(t)
    
    # First half - higher frequency
    audio_data[:len(t)//2] = 0.5 * np.sin(2 * np.pi * frequency_1 * t[:len(t)//2])
    
    # Second half - lower frequency  
    audio_data[len(t)//2:] = 0.5 * np.sin(2 * np.pi * frequency_2 * t[len(t)//2:])
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 0.05, len(audio_data))
    audio_data += noise
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        sf.write(temp_file.name, audio_data, sample_rate)
        return temp_file.name


def test_speaker_detection():
    """Test the speaker detection functionality"""
    print("🧪 Testing Speaker Diarization Service")
    print("=" * 50)
    
    # Create test audio
    print("📊 Creating test audio file...")
    test_audio_path = create_test_audio_file()
    
    try:
        # Read audio data
        with open(test_audio_path, "rb") as f:
            audio_data = f.read()
        
        print(f"✅ Test audio created: {len(audio_data)} bytes")
        
        # Test basic speaker detection
        print("\n🔍 Testing basic speaker detection...")
        diarization_service = SpeakerDiarizationService()
        
        if diarization_service.is_available:
            print("✅ Pyannote.audio is available")
        else:
            print("⚠️  Pyannote.audio not available, using fallback")
        
        speaker_result = diarization_service.detect_speakers(audio_data)
        
        print(f"🎯 Speaker Detection Results:")
        print(f"   - Number of speakers: {speaker_result.get('num_speakers', 'unknown')}")
        print(f"   - Method used: {speaker_result.get('method', 'unknown')}")
        print(f"   - Confidence: {speaker_result.get('confidence', 'unknown')}")
        print(f"   - Multi-speaker: {speaker_result.get('is_multi_speaker', False)}")
        
        if speaker_result.get('speaker_stats'):
            print(f"   - Speaker statistics:")
            for speaker, stats in speaker_result['speaker_stats'].items():
                print(f"     * {speaker}: {stats['total_speaking_time']}s ({stats['percentage_of_total']}%)")
        
        # Test enhanced audio service
        print("\n🚀 Testing enhanced transcription with speaker analysis...")
        enhanced_result = enhanced_audio_service.transcribe_with_speaker_analysis(audio_data)
        
        if 'error' in enhanced_result:
            print(f"❌ Enhanced transcription failed: {enhanced_result['error']}")
        else:
            print("✅ Enhanced transcription successful")
            print(f"   - Transcript length: {len(enhanced_result.get('transcript', ''))}")
            print(f"   - Speaker analysis included: {'speaker_analysis' in enhanced_result}")
            
            if 'speaker_analysis' in enhanced_result:
                speaker_info = enhanced_result['speaker_analysis']
                print(f"   - Speakers detected: {speaker_info.get('num_speakers', 'unknown')}")
                
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up test file
        if os.path.exists(test_audio_path):
            os.unlink(test_audio_path)
            print(f"🧹 Cleaned up test file")


def test_installation_requirements():
    """Test if all required packages are available"""
    print("📦 Testing Package Requirements")
    print("=" * 40)
    
    packages = [
        ("numpy", "numpy"),
        ("soundfile", "soundfile"),
        ("torch", "torch"),
        ("pyannote.audio", "pyannote.audio"),
        ("torchaudio", "torchaudio")
    ]
    
    for package_import, package_name in packages:
        try:
            __import__(package_import)
            print(f"✅ {package_name}: Available")
        except ImportError as e:
            print(f"❌ {package_name}: Not available - {e}")
    
    print()


if __name__ == "__main__":
    print("🎤 Speaker Diarization Test Suite")
    print("=" * 60)
    
    # Test package requirements first
    test_installation_requirements()
    
    # Test speaker detection functionality
    test_speaker_detection()
    
    print("\n🎯 Test Summary:")
    print("   - Run 'pip install -r requirements.txt' to install missing packages")
    print("   - Set HUGGINGFACE_TOKEN environment variable for full functionality")
    print("   - Use the new endpoints: /audio/speaker-detection and /audio/enhanced-evaluation-with-speakers")
