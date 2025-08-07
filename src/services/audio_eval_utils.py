from typing import Dict, List, Any


def build_enhanced_scorecard_payload(
    evaluation_result: Any,
    question_blocks: List[Dict],
    question_scorecard: Dict,
    speech_analysis: Dict | None = None,
) -> Dict:
    """Build a frontend-compatible payload from an enhanced evaluation.

    Returns a dict with top-level keys:
    - feedback: str
    - scorecard: List[Row]

    Each Row has:
    - category: str
    - feedback: { correct: Optional[str], wrong: Optional[str] }
    - score: float
    - max_score: int
    - pass_score: int
    """

    # Normalize raw scores to the question's scorecard schema
    raw_rows: List[Dict] = []

    for score_item in getattr(evaluation_result, "scores", []) or []:
        # Try to match an existing criterion from the question scorecard
        criterion = None
        for crit in question_scorecard.get("criteria", []):
            crit_name = (crit.get("name") or "").lower()
            item_name = (getattr(score_item, "criterion", "") or "").lower()
            if (
                crit_name in item_name
                or item_name in crit_name
                or any(word in crit_name for word in item_name.split())
            ):
                criterion = crit
                break

        if not criterion:
            # Reasonable defaults
            criterion = {
                "name": (getattr(score_item, "criterion", "").title() or "Criterion"),
                "min_score": 1,
                "max_score": 10,
                "pass_score": 6,
            }

        orig_max = int(criterion.get("max_score", 10)) or 10
        orig_pass = int(criterion.get("pass_score", max(1, int(0.6 * orig_max))))

        # Normalize raw score to a unified 10-point scale for display
        raw_score = float(getattr(score_item, "score", 0) or 0)
        base_scale = 5.0 if raw_score <= 5.0 else 10.0
        try:
            score_10 = max(0.0, min(10.0, (raw_score / base_scale) * 10.0))
        except Exception:
            score_10 = 0.0

        # Scale pass score proportionally to 10-point scale
        try:
            pass_10 = max(0.0, min(10.0, (orig_pass / float(orig_max)) * 10.0))
        except Exception:
            pass_10 = 6.0

        raw_rows.append(
            {
                "category": criterion["name"],
                "feedback_text": getattr(score_item, "feedback", ""),
                "score": round(score_10, 1),
                "max_score": 10,
                "pass_score": round(pass_10, 0),
            }
        )

    # Convert to frontend schema
    scorecard_rows: List[Dict] = []

    # Guardrail: multi-speaker detection alert row (if analysis indicates)
    try:
        if speech_analysis and speech_analysis.get("multi_speaker_suspected"):
            scorecard_rows.append(
                {
                    "category": "VALIDATION",
                    "feedback": {
                        "correct": None,
                        "wrong": "Multiple speakers detected in the recording. Please submit a single-speaker response for accurate evaluation.",
                    },
                    "score": 0,
                    "max_score": 0,
                    "pass_score": 0,
                }
            )
    except Exception:
        pass
    for row in raw_rows:
        scorecard_rows.append(
            {
                "category": row["category"],
                "feedback": {"correct": None, "wrong": row["feedback_text"]},
                "score": row["score"],
                "max_score": row["max_score"],
                "pass_score": row["pass_score"],
            }
        )

    # Compose summary feedback text
    tips = getattr(evaluation_result, "actionable_tips", []) or []
    tips_text = "\n".join([f"- {getattr(t, 'title', '')}: {getattr(t, 'description', '')}" for t in tips])

    scores = getattr(evaluation_result, "scores", []) or []
    scores_lines = [
        f"{getattr(s, 'criterion', '').title()}: {getattr(s, 'score', '')}/10 — {getattr(s, 'feedback', '')}"
        for s in scores
    ]

    summary_feedback = (
        f"Overall: {getattr(evaluation_result, 'overall_score', 0):.1f}/10\n\n"
        + "\n".join(scores_lines)
        + (f"\n\nActionable tips:\n{tips_text}" if tips_text else "")
    )

    return {"feedback": summary_feedback, "scorecard": scorecard_rows}


def format_fillers_for_prompt(fillers: List[Dict]) -> str:
    """Summarize filler usage for prompts."""
    if not fillers:
        return "No significant fillers detected."
    counts: Dict[str, int] = {}
    for f in fillers:
        w = (f.get("word") or "").lower().strip()
        counts[w] = counts.get(w, 0) + 1
    return "\n".join([f"- '{w}': {c} times" for w, c in counts.items()])


def create_enhanced_fallback_evaluation(
    transcript: str, analysis: Dict, question: str
) -> Dict:
    """Create a robust fallback evaluation (dict) preserving the public schema."""
    content_score = min(10, max(1, 6 + (len(transcript.split()) - 50) // 15))
    coherence_score = (
        analysis.get("coherence_analysis", {}).get("coherence_score", 8)
    )
    filler_penalty = min(3, analysis.get("filler_count", 0) // 2)
    wpm = analysis.get("speaking_rate_wpm", 150)
    rate_penalty = 1 if (wpm < 100 or wpm > 180) else 0
    clarity_score = max(1, min(10, 9 - filler_penalty - rate_penalty))
    confidence_score = 8
    language_score = analysis.get("language_analysis", {}).get("multilingual_score", 9)
    overall_score = (
        content_score + coherence_score + clarity_score + confidence_score + language_score
    ) / 5
    overall_score = max(6.0, min(9.0, overall_score))

    tips: List[Dict] = []
    if content_score < 6:
        tips.append(
            {
                "title": "Expand Content Depth",
                "description": f"Your response contained {analysis.get('word_count', 0)} words. Add specific examples and details to strengthen your answer.",
                "timestamp_ranges": [],
                "priority": 1,
            }
        )
    if analysis.get("filler_count", 0) > 3:
        tips.append(
            {
                "title": "Reduce Filler Words",
                "description": "Detected {} filler words. Practice pausing instead of using 'um', 'uh', etc.".format(
                    analysis.get("filler_count", 0)
                ),
                "timestamp_ranges": [],
                "priority": 2,
            }
        )
    coherence_issues = analysis.get("coherence_analysis", {}).get("coherence_issues", [])
    if coherence_issues:
        tips.append(
            {
                "title": "Improve Logical Flow",
                "description": "Coherence issues detected: {}".format("; ".join(coherence_issues[:2])),
                "timestamp_ranges": [],
                "priority": 3,
            }
        )
    if analysis.get("language_analysis", {}).get("is_multilingual", False):
        detected_langs = analysis.get("language_analysis", {}).get("detected_languages", [])
        tips.append(
            {
                "title": "Maintain Language Consistency",
                "description": "Multiple languages detected: {}. Stick to one language throughout.".format(
                    ", ".join(detected_langs)
                ),
                "timestamp_ranges": [],
                "priority": 4,
            }
        )
    repetition_score = analysis.get("repetition_analysis", {}).get("repetition_score", 10)
    if repetition_score < 7:
        tips.append(
            {
                "title": "Avoid Repetition",
                "description": "Excessive word/phrase repetition detected. Use synonyms and vary your expression.",
                "timestamp_ranges": [],
                "priority": 5,
            }
        )

    return {
        "scores": [
            {
                "criterion": "content_quality",
                "score": content_score,
                "feedback": "Content depth assessment based on {} words. Add more specific examples and details.".format(
                    analysis.get("word_count", 0)
                ),
                "transcript_references": [],
            },
            {
                "criterion": "coherence_flow",
                "score": coherence_score,
                "feedback": "Logical organization score: {}/10. Work on clear structure and transitions.".format(
                    coherence_score
                ),
                "transcript_references": [],
            },
            {
                "criterion": "communication_clarity",
                "score": clarity_score,
                "feedback": "Clarity assessment. {} fillers detected, speaking rate: {:.1f} WPM.".format(
                    analysis.get("filler_count", 0), analysis.get("speaking_rate_wpm", 0)
                ),
                "transcript_references": [],
            },
            {
                "criterion": "delivery_confidence",
                "score": confidence_score,
                "feedback": "Overall delivery clarity score: {}/10. Focus on confident, clear speech.".format(
                    confidence_score
                ),
                "transcript_references": [],
            },
            {
                "criterion": "language_consistency",
                "score": language_score,
                "feedback": "Language consistency score: {}/10. Maintain single language use.".format(
                    language_score
                ),
                "transcript_references": [],
            },
        ],
        "overall_score": round(overall_score, 1),
        "actionable_tips": tips[:4],
        "transcript": transcript,
        "speech_analysis": analysis,
        "duration_seconds": analysis.get("total_duration", 0),
    }


def create_fallback_evaluation(transcript: str, analysis: Dict) -> Dict:
    """Basic fallback evaluation as plain dict."""
    return {
        "scores": [
            {
                "criterion": "content",
                "score": 3,
                "feedback": "Response received and processed.",
                "transcript_references": [],
            }
        ],
        "overall_score": 3.0,
        "actionable_tips": [],
        "transcript": transcript,
        "speech_analysis": analysis,
        "duration_seconds": analysis.get("total_duration", 0),
    }


def generate_speech_recommendations(transcript: str, analysis: Dict, context: str | None = None) -> Dict:
    """Heuristic recommendations derived from analysis (no external calls)."""
    context_info = f"Context: {context}\n\n" if context else ""
    _ = context_info  # reserved for future prompt-driven implementations

    recommendations: Dict[str, Any] = {
        "priority_areas": [],
        "actionable_steps": {},
        "practice_exercises": [],
        "timeline": {},
    }

    scores = {
        "Clarity": analysis.get("overall_clarity_score", 5),
        "Coherence": analysis.get("coherence_analysis", {}).get("coherence_score", 5),
        "Language Consistency": analysis.get("language_analysis", {}).get("multilingual_score", 10),
        "Repetition Control": analysis.get("repetition_analysis", {}).get("repetition_score", 10),
    }

    for area, score in sorted(scores.items(), key=lambda x: x[1])[:3]:
        recommendations["priority_areas"].append(
            {
                "area": area,
                "current_score": score,
                "target_score": min(10, score + 2),
                "improvement_potential": "High" if score < 6 else ("Medium" if score < 8 else "Low"),
            }
        )

    if analysis.get("filler_count", 0) > 5:
        recommendations["actionable_steps"]["Reduce Fillers"] = [
            "Practice speaking with deliberate pauses instead of fillers",
            "Record yourself daily and count filler words",
            "Use the 'pause and breathe' technique when you feel a filler coming",
            "Practice presentations focusing on eliminating one filler type at a time",
            "Join a public speaking group for regular practice",
        ]

    if analysis.get("coherence_analysis", {}).get("coherence_score", 10) < 7:
        recommendations["actionable_steps"]["Improve Coherence"] = [
            "Use the PREP method: Point, Reason, Example, Point",
            "Practice with transition words: 'First', 'Moreover', 'Therefore', 'In conclusion'",
            "Outline your main points before speaking",
            "Practice connecting ideas with logical bridges",
            "Summarize key points at the end of each section",
        ]

    recommendations["practice_exercises"] = [
        "1-minute elevator pitches with a timer",
        "Read aloud focusing on clarity and pacing",
        "Record-and-review sessions focusing on fillers and pauses",
    ]

    recommendations["timeline"] = {
        "short_term": "Practice 10 minutes daily focusing on one improvement area",
        "long_term": "Deliver one 3-5 minute talk weekly and review progress",
    }

    return recommendations


