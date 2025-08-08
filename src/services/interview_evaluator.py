from typing import Dict, List
import json
from ..models.interview_rubric import InterviewRubric, InterviewEvaluation, CriterionScore, ActionableTip, RubricCriterion
from ..llm_service import LLMService
from api.utils.audio import audio_service

class InterviewEvaluator:
    def __init__(self):
        self.rubric = InterviewRubric.get_default_rubric()
        self.llm_service = LLMService()
    
    async def evaluate_audio_response(
        self, 
        audio_file_path: str, 
        question: str,
        context: Dict = None
    ) -> InterviewEvaluation:
        """Complete evaluation pipeline for audio interview response"""
        
        # 1. Load bytes & transcribe via OpenAI API wrapper (audio_service)
        with open(audio_file_path, 'rb') as f:
            audio_bytes = f.read()
        trans_res = audio_service.transcribe_with_analysis(audio_bytes)
        if trans_res.get('error'):
            raise ValueError(f"Transcription failed: {trans_res['error']}")

        transcription = {
            'text': trans_res.get('transcript', ''),
            'segments': trans_res.get('segments', []),
            'words': []  # not available from current wrapper
        }
        analysis = trans_res.get('analysis', {})
        fillers = analysis.get('fillers', [])
        duration_analysis = {
            'total_duration': analysis.get('total_duration', 0),
            'word_count': analysis.get('word_count', 0),
            'words_per_minute': analysis.get('speaking_rate_wpm', 0),
            'long_pauses': analysis.get('long_pauses', []),
            'target_duration': self.rubric.max_duration,
            'duration_score': self._score_duration(analysis.get('total_duration', 0))
        }
        
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
        
        # 6. Generate vocal delivery focused feedback list (JSON array of strings)
        vocal_feedback = await self._generate_vocal_feedback(
            transcription=transcription,
            fillers=fillers,
            duration_analysis=duration_analysis
        )

        # 7. Determine follow-up question if answer weak
        follow_up_question = self._maybe_follow_up(ai_evaluation)

        # 8. Create final evaluation enriched
        return InterviewEvaluation(
            scores=ai_evaluation["scores"],
            overall_score=ai_evaluation["overall_score"],
            actionable_tips=actionable_tips,
            transcript_highlights=self._highlight_transcript_issues(transcription, fillers),
            duration_analysis=duration_analysis,
            strengths=ai_evaluation.get("strengths", []),
            areas_for_improvement=ai_evaluation.get("areas_for_improvement", []),
            vocal_feedback=vocal_feedback,
            follow_up_question=follow_up_question
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

    async def _generate_vocal_feedback(self, transcription: Dict, fillers: List, duration_analysis: Dict) -> List[str]:
        """Generate up to 5 specific vocal delivery feedback points as JSON array strings."""
        system_prompt = """You are an expert vocal coach. You will be given transcript text (without timestamps here), filler data, and pacing stats. Produce 3-5 concise, constructive feedback points focusing ONLY on: Pace & Rhythm, Tone & Intonation, Clarity & Articulation, Filler Words usage, Volume/Projection (infer from clarity & pacing), and Pauses. Each point must: (a) Name the aspect, (b) Describe observed behavior, (c) Provide one actionable improvement suggestion, (d) Be encouraging. Return ONLY a JSON array of strings. No extra text."""

        filler_summary = self._format_fillers_for_prompt(fillers)
        user_message = (
            f"TRANSCRIPT:\n{transcription['text']}\n\n"
            f"DURATION: {duration_analysis['total_duration']:.1f}s | WPM: {duration_analysis['words_per_minute']:.1f} | LONG_PAUSES: {len(duration_analysis['long_pauses'])}\n"
            f"FILLERS:\n{filler_summary}\n"
            "Return JSON array now."
        )
        try:
            response = await self.llm_service.get_structured_response(
                system_prompt=system_prompt,
                user_message=user_message,
                response_format={"feedback": ["array of feedback strings"]}
            )
            # Accept either direct array or object wrapper
            if isinstance(response, list):
                return response[:5]
            if isinstance(response, dict):
                # heuristic: find first list value
                for v in response.values():
                    if isinstance(v, list):
                        return [str(x) for x in v][:5]
        except Exception:
            pass
        return []

    def _maybe_follow_up(self, ai_eval: Dict) -> str | None:
        """If overall or any key criterion weak, craft a follow-up coaching question."""
        overall = ai_eval.get("overall_score", 0)
        scores = ai_eval.get("scores", [])
        weak = [s for s in scores if s.get("score", 0) <= 2]
        if overall < 3 or weak:
            # target weakest criterion name(s)
            weak_names = ", ".join({s.get("criterion","") for s in weak}) or "areas above"
            return (
                f"Great effort. Let's refine your {weak_names}. Could you try again focusing on a clear core definition early, reducing fillers, and tightening structure?"
            )
        return None