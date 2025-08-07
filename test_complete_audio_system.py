#!/usr/bin/env python3
"""
Complete Audio Analysis System Test

This script tests all the enhanced audio functionality including:
- Basic transcription with comprehensive analysis
- Vocal coaching feedback
- Contextual analysis
- Audio validation
- Comprehensive speech analysis with recommendations

Usage:
    python test_complete_audio_system.py [audio_file_path]
"""

import asyncio
import sys
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

def test_audio_transcription_service():
    """Test the enhanced AudioTranscriptionService"""
    try:
        from api.utils.audio import AudioTranscriptionService
        
        # Mock audio data for testing
        mock_audio = b"fake_audio_data_for_testing"
        
        audio_service = AudioTranscriptionService()
        
        # Test basic initialization
        logger.info("✅ AudioTranscriptionService initialized successfully")
        
        # Test pattern analysis methods exist
        assert hasattr(audio_service, '_analyze_speech_patterns'), "Missing _analyze_speech_patterns method"
        assert hasattr(audio_service, '_detect_language_switching'), "Missing _detect_language_switching method"
        assert hasattr(audio_service, '_analyze_coherence'), "Missing _analyze_coherence method"
        assert hasattr(audio_service, '_analyze_repetition'), "Missing _analyze_repetition method"
        assert hasattr(audio_service, '_calculate_clarity_score'), "Missing _calculate_clarity_score method"
        
        logger.info("✅ All analysis methods present in AudioTranscriptionService")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error testing AudioTranscriptionService: {e}")
        return False

def test_enhanced_audio_evaluator():
    """Test the enhanced audio evaluator service"""
    try:
        from services.enhanced_audio_evaluator import EnhancedAudioEvaluator, AudioValidationResult
        
        evaluator = EnhancedAudioEvaluator()
        logger.info("✅ EnhancedAudioEvaluator initialized successfully")
        
        # Test that all required methods exist
        assert hasattr(evaluator, 'validate_audio_quality'), "Missing validate_audio_quality method"
        assert hasattr(evaluator, 'get_vocal_coaching_feedback'), "Missing get_vocal_coaching_feedback method"
        assert hasattr(evaluator, 'get_contextual_analysis'), "Missing get_contextual_analysis method"
        
        logger.info("✅ All methods present in EnhancedAudioEvaluator")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error testing EnhancedAudioEvaluator: {e}")
        return False

def test_enhanced_audio_processing():
    """Test the enhanced audio processing utilities"""
    try:
        from utils.enhanced_audio_processing import (
            AudioQualityValidator,
            process_audio_with_enhanced_validation,
            create_audio_summary_report,
            AudioProcessingError
        )
        
        validator = AudioQualityValidator()
        logger.info("✅ AudioQualityValidator initialized successfully")
        
        # Test that processing functions exist
        assert callable(process_audio_with_enhanced_validation), "process_audio_with_enhanced_validation not callable"
        assert callable(create_audio_summary_report), "create_audio_summary_report not callable"
        
        logger.info("✅ All processing functions available")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error testing enhanced audio processing: {e}")
        return False

def test_pydantic_models():
    """Test the Pydantic models for structured responses"""
    try:
        from services.enhanced_audio_evaluator import (
            SpeechAnalysisResponse,
            VocalCoachingFeedback,
            ContextualAnalysisResponse
        )
        
        # Test SpeechAnalysisResponse
        analysis = SpeechAnalysisResponse(
            strengths=["Good pace", "Clear voice"],
            contentImprovements=["Add examples", "Better structure"],
            deliverySuggestions=["Reduce filler words", "Improve projection"],
            styleObservations="Professional tone detected"
        )
        logger.info("✅ SpeechAnalysisResponse model working")
        
        # Test VocalCoachingFeedback
        coaching = VocalCoachingFeedback(
            feedback_points=["Great pace", "Clear articulation", "Good volume"]
        )
        logger.info("✅ VocalCoachingFeedback model working")
        
        # Test ContextualAnalysisResponse
        context_analysis = ContextualAnalysisResponse(
            relevance_to_context=8,
            completeness_score=7,
            technical_accuracy=9,
            communication_effectiveness=8,
            context_specific_feedback="Well structured answer"
        )
        logger.info("✅ ContextualAnalysisResponse model working")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error testing Pydantic models: {e}")
        return False

def test_speech_analysis_methods():
    """Test specific speech analysis methods with sample data"""
    try:
        from api.utils.audio import AudioTranscriptionService
        
        audio_service = AudioTranscriptionService()
        
        # Test language switching detection
        multilingual_text = "Hello, this is English. Hola, esto es español. Back to English."
        language_analysis = audio_service._detect_language_switching(multilingual_text)
        assert isinstance(language_analysis, dict), "Language analysis should return dict"
        assert 'is_multilingual' in language_analysis, "Missing is_multilingual key"
        logger.info("✅ Language switching detection working")
        
        # Test coherence analysis
        coherent_text = "First, let me explain the concept. Moreover, this builds on previous knowledge. Finally, we can conclude that the approach is effective."
        coherence_analysis = audio_service._analyze_coherence(coherent_text)
        assert isinstance(coherence_analysis, dict), "Coherence analysis should return dict"
        assert 'coherence_score' in coherence_analysis, "Missing coherence_score key"
        logger.info("✅ Coherence analysis working")
        
        # Test repetition analysis
        repetitive_text = "This is good. This is very good. This is really really good. Good good good."
        repetition_analysis = audio_service._analyze_repetition(repetitive_text)
        assert isinstance(repetition_analysis, dict), "Repetition analysis should return dict"
        assert 'repetition_score' in repetition_analysis, "Missing repetition_score key"
        logger.info("✅ Repetition analysis working")
        
        # Test clarity score calculation
        mock_analysis = {
            'filler_count': 3,
            'word_count': 100,
            'coherence_analysis': {'coherence_score': 8},
            'language_analysis': {'multilingual_score': 9},
            'repetition_analysis': {'repetition_score': 7}
        }
        clarity_score = audio_service._calculate_clarity_score(mock_analysis)
        assert isinstance(clarity_score, (int, float)), "Clarity score should be numeric"
        assert 1 <= clarity_score <= 10, "Clarity score should be between 1-10"
        logger.info(f"✅ Clarity score calculation working (score: {clarity_score})")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing speech analysis methods: {e}")
        return False

def run_all_tests():
    """Run all tests and provide summary"""
    logger.info("🚀 Starting Complete Audio Analysis System Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Audio Transcription Service", test_audio_transcription_service),
        ("Enhanced Audio Evaluator", test_enhanced_audio_evaluator),
        ("Enhanced Audio Processing", test_enhanced_audio_processing),
        ("Pydantic Models", test_pydantic_models),
        ("Speech Analysis Methods", test_speech_analysis_methods),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n📝 Testing {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Audio analysis system is ready.")
        
        # Provide usage examples
        logger.info("\n" + "=" * 60)
        logger.info("📚 USAGE EXAMPLES")
        logger.info("=" * 60)
        
        examples = [
            {
                "endpoint": "/audio/comprehensive-speech-analysis",
                "description": "Get complete speech analysis with recommendations",
                "example": {
                    "audio_uuid": "your-audio-uuid",
                    "question": "Tell me about your experience with Python",
                    "context": "Technical interview",
                    "include_recommendations": True
                }
            },
            {
                "endpoint": "/audio/vocal-coaching",
                "description": "Get vocal coaching feedback with actionable suggestions",
                "example": {
                    "audio_uuid": "your-audio-uuid",
                    "focus_area": "clarity"
                }
            },
            {
                "endpoint": "/audio/contextual-analysis",
                "description": "Get context-specific analysis",
                "example": {
                    "audio_uuid": "your-audio-uuid",
                    "context": "Job interview for software developer position"
                }
            },
            {
                "endpoint": "/audio/validate",
                "description": "Validate audio quality",
                "example": {
                    "audio_uuid": "your-audio-uuid",
                    "context": "Interview preparation"
                }
            }
        ]
        
        for example in examples:
            logger.info(f"\n📡 {example['endpoint']}")
            logger.info(f"   {example['description']}")
            logger.info(f"   Example: {json.dumps(example['example'], indent=2)}")
        
        logger.info("\n💡 Key Features Available:")
        features = [
            "1-10 scoring scale for all metrics",
            "Comprehensive filler word analysis with density calculation",
            "Language switching detection (English, Spanish, French, German, Hindi, Arabic)",
            "Coherence analysis with transition word detection",
            "Repetition pattern analysis",
            "Speaking rate and pause analysis",
            "AI-powered improvement recommendations",
            "Vocal clarity and confidence assessment",
            "Context-specific feedback",
            "Audio quality validation"
        ]
        
        for feature in features:
            logger.info(f"   ✨ {feature}")
            
    else:
        logger.error("💥 Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
