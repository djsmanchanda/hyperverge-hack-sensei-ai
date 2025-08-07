"""
Test script for enhanced audio functionality
"""
import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from services.enhanced_audio_evaluator import enhanced_audio_evaluator
from utils.enhanced_audio_processing import process_audio_with_enhanced_validation
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_audio_validation():
    """Test audio validation functionality"""
    print("Testing Enhanced Audio Functionality")
    print("=" * 50)
    
    # Test with minimal audio data (will likely fail validation)
    test_audio_data = b"fake_audio_data_for_testing"
    
    try:
        # Test 1: Audio validation
        print("\n1. Testing Audio Validation...")
        validation_result = enhanced_audio_evaluator.validate_audio_quality(test_audio_data)
        print(f"   Validation Result: {validation_result.is_valid}")
        print(f"   Errors: {validation_result.errors}")
        print(f"   Warnings: {validation_result.warnings}")
        
        # Test 2: Enhanced processing (will likely fail due to fake data)
        print("\n2. Testing Enhanced Processing...")
        try:
            processing_result = process_audio_with_enhanced_validation(
                test_audio_data, 
                context="Test interview question",
                validation_level="minimal"
            )
            print(f"   Processing Status: {processing_result.get('processing_status')}")
        except Exception as e:
            print(f"   Expected processing error: {str(e)}")
        
        # Test 3: Prompt templates
        print("\n3. Testing Prompt Templates...")
        from services.enhanced_audio_evaluator import (
            VOCAL_COACHING_SYSTEM_INSTRUCTION,
            create_system_prompt_template,
            DEFAULT_SYSTEM_PROMPT
        )
        
        print("   ✓ Vocal coaching prompt loaded")
        print("   ✓ System prompt template function available")
        print("   ✓ Default system prompt loaded")
        
        contextual_prompt = create_system_prompt_template("practice presentation skills")
        print(f"   ✓ Contextual prompt generated (length: {len(contextual_prompt)} chars)")
        
        # Test 4: Pydantic models
        print("\n4. Testing Pydantic Models...")
        from services.enhanced_audio_evaluator import (
            SpeechAnalysisResponse,
            VocalCoachingFeedback
        )
        
        # Test SpeechAnalysisResponse model
        test_analysis = SpeechAnalysisResponse(
            strengths=["Good pace", "Clear articulation"],
            contentImprovements=["Add more examples", "Better structure"],
            deliverySuggestions=["Speak louder", "Reduce filler words"],
            styleObservations="Professional tone detected"
        )
        print(f"   ✓ SpeechAnalysisResponse model: {len(test_analysis.strengths)} strengths")
        
        test_feedback = VocalCoachingFeedback(
            feedback_points=["Great pace", "Clear articulation", "Good volume"]
        )
        print(f"   ✓ VocalCoachingFeedback model: {len(test_feedback.feedback_points)} points")
        
        # Test 5: Audio processing utilities
        print("\n5. Testing Audio Processing Utilities...")
        from utils.enhanced_audio_processing import (
            calculate_audio_quality_score,
            generate_audio_recommendations,
            create_audio_summary_report
        )
        
        # Test with mock data
        mock_stats = {
            "duration": 45.0,
            "word_count": 120,
            "filler_count": 3,
            "speaking_rate": 145.0
        }
        
        quality_score = calculate_audio_quality_score(mock_stats, {"word_count": 120})
        print(f"   ✓ Quality score calculation: {quality_score:.2f}")
        
        print("\n✅ All tests completed successfully!")
        print("\nNext Steps:")
        print("- Upload a real audio file to test full functionality")
        print("- Test API endpoints with actual audio data")
        print("- Verify integration with frontend components")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_audio_validation())
