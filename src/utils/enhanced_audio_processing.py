"""
Enhanced audio processing utilities with improved error handling and validation
"""
import tempfile
import os
from typing import Dict, List, Optional, Tuple
from api.utils.audio import audio_service
from services.enhanced_audio_evaluator import enhanced_audio_evaluator
import logging

logger = logging.getLogger(__name__)

class AudioProcessingError(Exception):
    """Custom exception for audio processing errors"""
    pass

class AudioQualityValidator:
    """Validates audio quality before processing"""
    
    @staticmethod
    def validate_file_format(file_path: str) -> bool:
        """Validate if file is in supported audio format"""
        supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
        return any(file_path.lower().endswith(fmt) for fmt in supported_formats)
    
    @staticmethod
    def validate_file_size(file_path: str, max_size_mb: int = 50) -> bool:
        """Validate file size"""
        try:
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            return size_mb <= max_size_mb
        except OSError:
            return False

def process_audio_with_enhanced_validation(
    audio_data: bytes, 
    context: Optional[str] = None,
    validation_level: str = "standard"
) -> Dict:
    """
    Process audio with enhanced validation and guardrails
    
    Args:
        audio_data: Raw audio bytes
        context: Optional context for the audio (e.g., interview question)
        validation_level: "minimal", "standard", or "strict"
    
    Returns:
        Dictionary with transcription, analysis, and validation results
    """
    try:
        # Step 1: Basic audio validation
        validation_result = enhanced_audio_evaluator.validate_audio_quality(audio_data, context)
        
        if validation_level == "strict" and not validation_result.is_valid:
            raise AudioProcessingError(f"Audio validation failed: {'; '.join(validation_result.errors)}")
        
        # Step 2: Get transcription and analysis
        transcription_result = audio_service.transcribe_with_analysis(audio_data)
        
        if "error" in transcription_result:
            raise AudioProcessingError(f"Transcription failed: {transcription_result['error']}")
        
        # Step 3: Enhanced quality scoring
        quality_score = calculate_audio_quality_score(
            validation_result.audio_stats, 
            transcription_result["analysis"]
        )
        
        # Step 4: Generate recommendations based on quality
        recommendations = generate_audio_recommendations(
            validation_result, 
            transcription_result["analysis"]
        )
        
        return {
            "transcript": transcription_result["transcript"],
            "analysis": transcription_result["analysis"],
            "validation": {
                "is_valid": validation_result.is_valid,
                "errors": validation_result.errors,
                "warnings": validation_result.warnings,
                "audio_stats": validation_result.audio_stats
            },
            "quality_score": quality_score,
            "recommendations": recommendations,
            "processing_status": "success"
        }
        
    except AudioProcessingError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in audio processing: {str(e)}")
        raise AudioProcessingError(f"Audio processing failed: {str(e)}")

def calculate_audio_quality_score(audio_stats: Optional[Dict], analysis: Dict) -> float:
    """Calculate overall audio quality score (0-1)"""
    if not audio_stats:
        return 0.5  # Neutral score when stats unavailable
    
    score_components = []
    
    # Duration score (optimal: 30-120 seconds)
    duration = audio_stats.get("duration", 0)
    if 30 <= duration <= 120:
        duration_score = 1.0
    elif 15 <= duration <= 180:
        duration_score = 0.8
    elif 5 <= duration <= 300:
        duration_score = 0.6
    else:
        duration_score = 0.3
    score_components.append(duration_score)
    
    # Speaking rate score (optimal: 120-160 WPM)
    wpm = audio_stats.get("speaking_rate", 0)
    if 120 <= wpm <= 160:
        rate_score = 1.0
    elif 100 <= wpm <= 180:
        rate_score = 0.8
    elif 80 <= wpm <= 200:
        rate_score = 0.6
    else:
        rate_score = 0.4
    score_components.append(rate_score)
    
    # Filler word score
    filler_count = audio_stats.get("filler_count", 0)
    word_count = audio_stats.get("word_count", 1)
    filler_ratio = filler_count / word_count if word_count > 0 else 0
    if filler_ratio <= 0.05:
        filler_score = 1.0
    elif filler_ratio <= 0.1:
        filler_score = 0.8
    elif filler_ratio <= 0.2:
        filler_score = 0.6
    else:
        filler_score = 0.3
    score_components.append(filler_score)
    
    # Content score (based on word count and transcript length)
    if word_count >= 50:
        content_score = 1.0
    elif word_count >= 25:
        content_score = 0.8
    elif word_count >= 10:
        content_score = 0.6
    else:
        content_score = 0.3
    score_components.append(content_score)
    
    return sum(score_components) / len(score_components)

def generate_audio_recommendations(validation_result, analysis: Dict) -> List[Dict]:
    """Generate specific recommendations for audio improvement"""
    recommendations = []
    
    if not validation_result.audio_stats:
        return [{"type": "error", "message": "Unable to analyze audio quality"}]
    
    stats = validation_result.audio_stats
    
    # Duration recommendations
    duration = stats.get("duration", 0)
    if duration < 15:
        recommendations.append({
            "type": "duration",
            "severity": "warning",
            "message": f"Recording is quite short ({duration:.1f}s). Consider providing more detailed responses (30-60s recommended).",
            "action": "Expand your response with specific examples and details"
        })
    elif duration > 180:
        recommendations.append({
            "type": "duration", 
            "severity": "info",
            "message": f"Recording is quite long ({duration:.1f}s). Practice being more concise while maintaining detail.",
            "action": "Focus on key points and eliminate unnecessary repetition"
        })
    
    # Speaking rate recommendations
    wpm = stats.get("speaking_rate", 0)
    if wpm < 100:
        recommendations.append({
            "type": "pace",
            "severity": "warning", 
            "message": f"Speaking pace is slow ({wpm:.0f} WPM). Try to speak more confidently and reduce long pauses.",
            "action": "Practice speaking at 120-150 WPM for optimal comprehension"
        })
    elif wpm > 180:
        recommendations.append({
            "type": "pace",
            "severity": "warning",
            "message": f"Speaking pace is fast ({wpm:.0f} WPM). Slow down to ensure clarity and allow processing time.",
            "action": "Practice pausing between sentences and speaking more deliberately"
        })
    
    # Filler word recommendations
    filler_count = stats.get("filler_count", 0)
    if filler_count > 5:
        recommendations.append({
            "type": "fillers",
            "severity": "info",
            "message": f"{filler_count} filler words detected. Work on reducing 'um', 'uh', and similar hesitations.",
            "action": "Practice pausing silently instead of using filler words"
        })
    
    # Content recommendations
    word_count = stats.get("word_count", 0)
    if word_count < 25:
        recommendations.append({
            "type": "content",
            "severity": "warning",
            "message": f"Response seems brief ({word_count} words). Consider providing more comprehensive answers.",
            "action": "Use the STAR method (Situation, Task, Action, Result) to structure fuller responses"
        })
    
    # Add validation warnings as recommendations
    for warning in validation_result.warnings:
        recommendations.append({
            "type": "quality",
            "severity": "info", 
            "message": warning,
            "action": "Consider re-recording in a quieter environment with better microphone positioning"
        })
    
    return recommendations

def create_audio_summary_report(processing_result: Dict) -> Dict:
    """Create a comprehensive summary report of audio analysis"""
    analysis = processing_result.get("analysis", {})
    validation = processing_result.get("validation", {})
    
    return {
        "overall_quality": processing_result.get("quality_score", 0.5),
        "key_metrics": {
            "duration": f"{analysis.get('total_duration', 0):.1f}s",
            "word_count": analysis.get('word_count', 0),
            "speaking_rate": f"{analysis.get('speaking_rate_wpm', 0):.0f} WPM",
            "filler_words": analysis.get('filler_count', 0),
            "long_pauses": len(analysis.get('long_pauses', []))
        },
        "strengths": extract_strengths(analysis, validation),
        "areas_for_improvement": extract_improvements(processing_result.get("recommendations", [])),
        "next_steps": generate_next_steps(processing_result.get("recommendations", []))
    }

def extract_strengths(analysis: Dict, validation: Dict) -> List[str]:
    """Extract positive aspects from audio analysis"""
    strengths = []
    
    # Good speaking rate
    wpm = analysis.get('speaking_rate_wpm', 0)
    if 120 <= wpm <= 160:
        strengths.append(f"Excellent speaking pace ({wpm:.0f} WPM)")
    
    # Low filler usage
    filler_count = analysis.get('filler_count', 0)
    if filler_count <= 3:
        strengths.append("Minimal use of filler words")
    
    # Good duration
    duration = analysis.get('total_duration', 0)
    if 30 <= duration <= 120:
        strengths.append("Well-timed response length")
    
    # Clear speech (inferred from successful transcription)
    if validation.get('is_valid', False):
        strengths.append("Clear and audible speech")
    
    if not strengths:
        strengths.append("Audio was successfully processed and analyzed")
    
    return strengths

def extract_improvements(recommendations: List[Dict]) -> List[str]:
    """Extract improvement areas from recommendations"""
    improvements = []
    
    for rec in recommendations:
        if rec.get("severity") in ["warning", "error"]:
            improvements.append(rec.get("message", ""))
    
    return improvements[:3]  # Limit to top 3

def generate_next_steps(recommendations: List[Dict]) -> List[str]:
    """Generate actionable next steps"""
    next_steps = []
    
    for rec in recommendations:
        action = rec.get("action")
        if action and action not in next_steps:
            next_steps.append(action)
    
    if not next_steps:
        next_steps = [
            "Continue practicing regular speech recording",
            "Focus on maintaining consistent pacing",
            "Work on providing structured, detailed responses"
        ]
    
    return next_steps[:4]  # Limit to top 4 actions
