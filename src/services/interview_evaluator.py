from typing import Dict, List
import json
from .transcription_service import TranscriptionService
from ..models.interview_rubric import InterviewRubric, InterviewEvaluation, CriterionScore, ActionableTip, RubricCriterion
from ..llm.llm_service import LLMService

class InterviewEvaluator:
    def __init__(self):
        self.transcription_service = TranscriptionService()
        self.rubric = InterviewRubric.get_default_rubric()
        self.llm_service = LLMService()
    
    async def evaluate_audio_response(
        self, 
        audio_file_path: str, 
        question: str,
        context: Dict = None
    ) -> InterviewEvaluation:
        """Complete evaluation pipeline for audio interview response"""
        
        # 1. Transcribe with timestamps
        transcription = self.transcription_service.transcribe_with_timestamps(audio_file_path)
        
        # 2. Detect fillers and speech patterns
        fillers = self.transcription_service.detect_fillers(transcription["words"])
        
        # 3. Analyze duration and pacing
        duration_analysis = self._analyze_duration_and_pacing(transcription)
        
        # 4. Get AI evaluation using enhanced prompt
        ai_evaluation = await self._get_ai_evaluation(
            question=question,
            transcript=transcription["text"],
            segments=transcription["segments"],
            fillers=fillers,
            duration_analysis=duration_analysis,
            context=context
        )
        
        # 5. Generate actionable tips
        actionable_tips = self._generate_actionable_tips(
            ai_evaluation,
            transcription,
            fillers,
            duration_analysis
        )
        
        # 6. Create final evaluation
        return InterviewEvaluation(
            scores=ai_evaluation["scores"],
            overall_score=ai_evaluation["overall_score"],
            actionable_tips=actionable_tips,
            transcript_highlights=self._highlight_transcript_issues(transcription, fillers),
            duration_analysis=duration_analysis
        )
    
    def _analyze_duration_and_pacing(self, transcription: Dict) -> Dict:
        """Analyze speech duration and pacing"""
        total_duration = transcription["segments"][-1]["end"] if transcription["segments"] else 0
        word_count = len(transcription["words"])
        
        # Calculate speaking rate (words per minute)
        wpm = (word_count / total_duration) * 60 if total_duration > 0 else 0
        
        # Identify long pauses
        long_pauses = []
        for i in range(1, len(transcription["segments"])):
            gap = transcription["segments"][i]["start"] - transcription["segments"][i-1]["end"]
            if gap > 2.0:  # Pause longer than 2 seconds
                long_pauses.append({
                    "start": transcription["segments"][i-1]["end"],
                    "end": transcription["segments"][i]["start"],
                    "duration": gap
                })
        
        return {
            "total_duration": total_duration,
            "word_count": word_count,
            "words_per_minute": wpm,
            "long_pauses": long_pauses,
            "target_duration": self.rubric.max_duration,
            "duration_score": self._score_duration(total_duration)
        }
    
    def _score_duration(self, duration: float) -> int:
        """Score based on duration relative to target"""
        target = self.rubric.max_duration
        if duration <= target:
            return 5
        elif duration <= target * 1.2:
            return 4
        elif duration <= target * 1.5:
            return 3
        elif duration <= target * 2:
            return 2
        else:
            return 1
    
    async def _get_ai_evaluation(
        self,
        question: str,
        transcript: str,
        segments: List,
        fillers: List,
        duration_analysis: Dict,
        context: Dict = None
    ) -> Dict:
        """Get AI evaluation using enhanced rubric-based prompt"""
        
        system_prompt = f"""You are an expert interview coach evaluating a candidate's audio response.

EVALUATION CRITERIA (Score 1-5 for each):

1. CONTENT (1-5): Accuracy, depth, and relevance of information
2. STRUCTURE (1-5): Logical organization and flow of ideas  
3. CLARITY (1-5): Clear communication and word choice
4. DELIVERY (1-5): Confidence, pacing, and minimal fillers

RESPONSE DATA:
Question: {question}
Transcript: {transcript}
Duration: {duration_analysis['total_duration']:.1f}s (target: {duration_analysis['target_duration']}s)
Speaking Rate: {duration_analysis['words_per_minute']:.1f} WPM
Filler Count: {len(fillers)}
Long Pauses: {len(duration_analysis['long_pauses'])}

FILLER ANALYSIS:
{self._format_fillers_for_prompt(fillers)}

Provide scores and specific feedback for each criterion. Reference specific parts of the transcript when giving feedback."""

        response = await self.llm_service.get_structured_response(
            system_prompt=system_prompt,
            user_message="Evaluate this interview response according to the rubric.",
            response_format={
                "scores": [
                    {
                        "criterion": "content",
                        "score": "int",
                        "feedback": "string",
                        "transcript_references": ["array of specific quotes"]
                    }
                ],
                "overall_score": "float",
                "strengths": ["array"],
                "areas_for_improvement": ["array"]
            }
        )
        
        return response
    
    def _format_fillers_for_prompt(self, fillers: List) -> str:
        """Format filler analysis for AI prompt"""
        if not fillers:
            return "No significant fillers detected."
        
        filler_summary = {}
        for filler in fillers:
            word = filler["word"].lower().strip()
            if word in filler_summary:
                filler_summary[word] += 1
            else:
                filler_summary[word] = 1
        
        return "\n".join([f"- '{word}': {count} times" for word, count in filler_summary.items()])
    
    def _generate_actionable_tips(
        self,
        ai_evaluation: Dict,
        transcription: Dict,
        fillers: List,
        duration_analysis: Dict
    ) -> List[ActionableTip]:
        """Generate 2-3 specific actionable tips with transcript references"""
        
        tips = []
        
        # Tip 1: Content/Structure improvement
        lowest_score_criterion = min(ai_evaluation["scores"], key=lambda x: x["score"])
        if lowest_score_criterion["score"] <= 3:
            tips.append(ActionableTip(
                title=f"Improve {lowest_score_criterion['criterion'].title()}",
                description=lowest_score_criterion["feedback"],
                transcript_lines=lowest_score_criterion.get("transcript_references", []),
                priority=1
            ))
        
        # Tip 2: Filler reduction
        if len(fillers) > 3:
            filler_timestamps = [{"start": f["start"], "end": f["end"], "word": f["word"]} for f in fillers[:3]]
            tips.append(ActionableTip(
                title="Reduce Filler Words",
                description=f"Replace {len(fillers)} filler words with pauses or specific examples. Most frequent: {self._get_most_common_filler(fillers)}",
                timestamp_ranges=filler_timestamps,
                priority=2
            ))
        
        # Tip 3: Pacing/Duration
        if duration_analysis["total_duration"] > self.rubric.max_duration * 1.2:
            tips.append(ActionableTip(
                title="Tighten Your Response",
                description=f"Response was {duration_analysis['total_duration']:.1f}s (target: {self.rubric.max_duration}s). Focus on key points and practice concise explanations.",
                priority=3
            ))
        elif duration_analysis["words_per_minute"] < 120:
            tips.append(ActionableTip(
                title="Increase Speaking Pace", 
                description=f"Speaking rate of {duration_analysis['words_per_minute']:.1f} WPM is slow. Practice speaking more confidently and reducing long pauses.",
                priority=3
            ))
        
        return tips[:3]  # Return max 3 tips
    
    def _get_most_common_filler(self, fillers: List) -> str:
        """Get the most commonly used filler word"""
        if not fillers:
            return ""
        
        filler_counts = {}
        for filler in fillers:
            word = filler["word"].lower().strip()
            filler_counts[word] = filler_counts.get(word, 0) + 1
        
        return max(filler_counts.items(), key=lambda x: x[1])[0]
    
    def _highlight_transcript_issues(self, transcription: Dict, fillers: List) -> List[Dict]:
        """Create highlights for transcript display"""
        highlights = []
        
        # Highlight filler words
        for filler in fillers:
            highlights.append({
                "start": filler["start"],
                "end": filler["end"],
                "text": filler["word"],
                "type": "filler",
                "severity": "warning"
            })
        
        return highlights