"""
Enhanced Audio Evaluation Service with improved prompts and guardrails
"""
from typing import Dict, List, Optional, Union
import json
import re
from dataclasses import dataclass
from api.utils.audio import audio_service
from api.settings import settings
from api.llm import run_llm_with_instructor
from api.config import openai_plan_to_model_name
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

# System instruction for vocal coaching analysis
VOCAL_COACHING_SYSTEM_INSTRUCTION = """You are an expert vocal coach and speech analysis specialist. Listen to this audio recording carefully. 
Provide detailed, constructive feedback on the speaker's vocal delivery. Focus specifically on:

- Pace and Rhythm: Is it too fast, too slow, or varied appropriately?
- Tone and Intonation: Does the tone match the message? Is there enough vocal variety?
- Clarity and Articulation: Are words spoken clearly? Are there issues with mumbling?
- Filler Words: Identify usage of 'um', 'uh', 'like', etc.
- Volume and Projection: Is the speaker audible and confident?
- Pauses: Are pauses used effectively, or are they awkward?

Provide your feedback as a JSON array of strings, where each string is a distinct feedback point. Aim for up to 5 feedback points. 
Be encouraging and actionable. 
Return ONLY the JSON array (e.g., ["Feedback point 1.", "Feedback point 2."]). Do not include any other text, preamble, or sign-off."""

def create_system_prompt_template(user_prompt: str) -> str:
    """Create a system prompt template for contextual speech analysis"""
    return f"""
You're a professional speech coach. Analyze this transcript in the context of: "{user_prompt}". Provide:
1. 3 specific strengths
2. 3 very specific content improvements
3. 3 vocal delivery suggestions
4. Key speaking style observations

Return JSON ONLY
Please do not include numerical citations or bracketed numbers (e.g., [1], [2, 3]). If a feedback point refers to specific words from the transcript, cite them directly, for example: "Your point about 'xyz' was strong" becomes "Your point about 'xyz' was strong ["xyz"]". Only cite direct short quotes from the transcript in this format when highly relevant. Otherwise, provide no citation.
Keep the responses casual and friendly and light!
Do not include markdown or text formatting
Format response as JSON with following format. No extra information or text. just the JSON
{{
  "strengths": string[],
  "contentImprovements": string[],
  "deliverySuggestions": string[],
  "styleObservations": string
}}"""

DEFAULT_SYSTEM_PROMPT = """
You're a professional speech coach. Analyze this transcript and provide:
1. 3 strengths
2. 3 content improvements
3. 3 delivery suggestions
4. Style observations

Return JSON ONLY
Please do not include numerical citations or bracketed numbers (e.g., [1], [2, 3]). If a feedback point refers to specific words from the transcript, cite them directly, for example: "Your point about 'xyz' was strong" becomes "Your point about 'xyz' was strong ["xyz"]". Only cite direct short quotes from the transcript in this format when highly relevant. Otherwise, provide no citation.
Keep the responses casual and friendly and light!
Do not include markdown or text formatting
Format response as JSON with following format. No extra information or text. just the JSON
{
  "strengths": string[],
  "contentImprovements": string[],
  "deliverySuggestions": string[],
  "styleObservations": string
}"""

# Pydantic models for structured responses
class SpeechAnalysisResponse(BaseModel):
    strengths: List[str] = Field(description="3 specific strengths in the speech")
    contentImprovements: List[str] = Field(description="3 specific content improvements")
    deliverySuggestions: List[str] = Field(description="3 vocal delivery suggestions")
    styleObservations: str = Field(description="Key speaking style observations")

class VocalCoachingFeedback(BaseModel):
    feedback_points: List[str] = Field(description="Up to 5 specific feedback points about vocal delivery")

@dataclass
class AudioValidationResult:
    """Result of audio validation checks"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    audio_stats: Optional[Dict]

class EnhancedAudioEvaluator:
    """Enhanced audio evaluator with improved guardrails and prompts"""
    
    def __init__(self):
        self.min_duration = 3.0  # Minimum 3 seconds
        self.max_duration = 300.0  # Maximum 5 minutes
        self.min_words = 5  # Minimum word count
        self.max_filler_ratio = 0.3  # Maximum 30% filler words
        
    def validate_audio_quality(self, audio_data: bytes, context: Optional[str] = None) -> AudioValidationResult:
        """
        Validate audio quality and content before analysis
        """
        errors = []
        warnings = []
        audio_stats = None
        
        try:
            # Get transcription and analysis
            transcription_result = audio_service.transcribe_with_analysis(audio_data)
            
            if "error" in transcription_result:
                errors.append(f"Transcription failed: {transcription_result['error']}")
                return AudioValidationResult(False, errors, warnings, None)
            
            analysis = transcription_result["analysis"]
            transcript = transcription_result["transcript"]
            
            audio_stats = {
                "duration": analysis["total_duration"],
                "word_count": analysis["word_count"],
                "filler_count": analysis["filler_count"],
                "speaking_rate": analysis["speaking_rate_wpm"],
                "transcript_length": len(transcript.strip())
            }
            
            # Duration checks
            if analysis["total_duration"] < self.min_duration:
                errors.append(f"Audio too short: {analysis['total_duration']:.1f}s (minimum {self.min_duration}s)")
            elif analysis["total_duration"] > self.max_duration:
                warnings.append(f"Audio very long: {analysis['total_duration']:.1f}s (recommended max {self.max_duration}s)")
            
            # Content checks
            if analysis["word_count"] < self.min_words:
                errors.append(f"Insufficient speech content: {analysis['word_count']} words (minimum {self.min_words})")
            
            # Check if transcript is mostly silence or gibberish
            if len(transcript.strip()) < 10:
                errors.append("Audio appears to contain no meaningful speech")
            
            # Filler word ratio check
            if analysis["word_count"] > 0:
                filler_ratio = analysis["filler_count"] / analysis["word_count"]
                if filler_ratio > self.max_filler_ratio:
                    warnings.append(f"High filler word ratio: {filler_ratio:.1%} (recommended < {self.max_filler_ratio:.1%})")
            
            # Speaking rate checks
            if analysis["speaking_rate_wpm"] < 80:
                warnings.append(f"Very slow speaking rate: {analysis['speaking_rate_wpm']:.0f} WPM")
            elif analysis["speaking_rate_wpm"] > 200:
                warnings.append(f"Very fast speaking rate: {analysis['speaking_rate_wpm']:.0f} WPM")
            
            # Check for audio quality indicators
            if analysis["long_pauses"] and len(analysis["long_pauses"]) > 3:
                warnings.append(f"Multiple long pauses detected ({len(analysis['long_pauses'])} pauses > 2s)")
                
        except Exception as e:
            logger.error(f"Error in audio validation: {str(e)}")
            errors.append(f"Audio validation failed: {str(e)}")
        
        is_valid = len(errors) == 0
        return AudioValidationResult(is_valid, errors, warnings, audio_stats)
    
    async def get_vocal_coaching_feedback(self, audio_data: bytes) -> Dict:
        """
        Get vocal coaching feedback using the enhanced system instruction
        """
        try:
            # Validate audio first
            validation = self.validate_audio_quality(audio_data)
            if not validation.is_valid:
                return {
                    "error": "Audio validation failed",
                    "validation_errors": validation.errors,
                    "validation_warnings": validation.warnings
                }
            
            # Get transcription
            transcription_result = audio_service.transcribe_with_analysis(audio_data)
            transcript = transcription_result["transcript"]
            analysis = transcription_result["analysis"]
            
            # Enhanced prompt with audio analysis data
            enhanced_prompt = f"""
{VOCAL_COACHING_SYSTEM_INSTRUCTION}

AUDIO ANALYSIS DATA:
- Duration: {analysis['total_duration']:.1f} seconds
- Word Count: {analysis['word_count']} words
- Speaking Rate: {analysis['speaking_rate_wpm']:.1f} WPM
- Filler Words: {analysis['filler_count']} ({', '.join([f['word'] for f in analysis['fillers'][:5]])})
- Long Pauses: {len(analysis['long_pauses'])} pauses > 2 seconds

TRANSCRIPT:
{transcript}

Focus your feedback on the vocal delivery aspects mentioned above. Be specific and actionable.
"""
            
            response = await run_llm_with_instructor(
                api_key=settings.openai_api_key,
                model=openai_plan_to_model_name["text"],
                messages=[
                    {"role": "system", "content": enhanced_prompt},
                    {"role": "user", "content": "Analyze this speech and provide vocal coaching feedback."}
                ],
                response_model=VocalCoachingFeedback,
                max_completion_tokens=2048,
            )
            
            return {
                "feedback_points": response.feedback_points,
                "audio_stats": validation.audio_stats,
                "validation_warnings": validation.warnings,
                "transcript": transcript,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Error in vocal coaching feedback: {str(e)}")
            return {
                "error": f"Failed to generate vocal coaching feedback: {str(e)}",
                "fallback_feedback": self._get_fallback_vocal_feedback(validation.audio_stats if validation else None)
            }
    
    async def get_contextual_speech_analysis(self, audio_data: bytes, user_prompt: str) -> Dict:
        """
        Get contextual speech analysis based on user's specific prompt/context
        """
        try:
            # Validate audio first
            validation = self.validate_audio_quality(audio_data, user_prompt)
            if not validation.is_valid:
                return {
                    "error": "Audio validation failed",
                    "validation_errors": validation.errors,
                    "validation_warnings": validation.warnings
                }
            
            # Get transcription
            transcription_result = audio_service.transcribe_with_analysis(audio_data)
            transcript = transcription_result["transcript"]
            analysis = transcription_result["analysis"]
            
            # Use contextual system prompt
            system_prompt = create_system_prompt_template(user_prompt) if user_prompt else DEFAULT_SYSTEM_PROMPT
            
            # Enhanced user message with context
            user_message = f"""
CONTEXT: {user_prompt}

TRANSCRIPT: {transcript}

SPEECH METRICS:
- Duration: {analysis['total_duration']:.1f}s
- Words: {analysis['word_count']} 
- Rate: {analysis['speaking_rate_wpm']:.1f} WPM
- Fillers: {analysis['filler_count']}
- Pauses: {len(analysis['long_pauses'])} long pauses

Analyze this speech in the given context and provide structured feedback.
"""
            
            response = await run_llm_with_instructor(
                api_key=settings.openai_api_key,
                model=openai_plan_to_model_name["text"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                response_model=SpeechAnalysisResponse,
                max_completion_tokens=3072,
            )
            
            return {
                "strengths": response.strengths,
                "contentImprovements": response.contentImprovements,
                "deliverySuggestions": response.deliverySuggestions,
                "styleObservations": response.styleObservations,
                "audio_stats": validation.audio_stats,
                "validation_warnings": validation.warnings,
                "transcript": transcript,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Error in contextual speech analysis: {str(e)}")
            return {
                "error": f"Failed to generate speech analysis: {str(e)}",
                "fallback_analysis": self._get_fallback_speech_analysis(validation.audio_stats if validation else None)
            }
    
    def _get_fallback_vocal_feedback(self, audio_stats: Optional[Dict]) -> List[str]:
        """Generate fallback vocal feedback when AI analysis fails"""
        if not audio_stats:
            return ["Audio processed successfully. Please try again for detailed feedback."]
        
        feedback = []
        
        # Duration feedback
        if audio_stats["duration"] < 10:
            feedback.append("Consider speaking for a longer duration to provide more comprehensive content.")
        elif audio_stats["duration"] > 120:
            feedback.append("Great length! Make sure to maintain engagement throughout longer responses.")
        
        # Speaking rate feedback
        if audio_stats["speaking_rate"] < 100:
            feedback.append("Consider increasing your speaking pace slightly for better engagement.")
        elif audio_stats["speaking_rate"] > 180:
            feedback.append("Consider slowing down slightly to ensure clarity and comprehension.")
        
        # Filler words feedback
        if audio_stats["filler_count"] > 5:
            feedback.append("Try to reduce filler words by pausing instead of using 'um', 'uh', etc.")
        
        return feedback
    
    def _get_fallback_speech_analysis(self, audio_stats: Optional[Dict]) -> Dict:
        """Generate fallback speech analysis when AI analysis fails"""
        return {
            "strengths": ["Audio was successfully processed", "Speech was clearly audible", "Good effort in providing a response"],
            "contentImprovements": ["Focus on key points", "Provide specific examples", "Structure your response clearly"],
            "deliverySuggestions": ["Practice pacing", "Use confident tone", "Minimize filler words"],
            "styleObservations": "Basic speech analysis completed"
        }

# Global instance
enhanced_audio_evaluator = EnhancedAudioEvaluator()
