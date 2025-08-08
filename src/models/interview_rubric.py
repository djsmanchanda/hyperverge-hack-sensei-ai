from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum

class RubricCriterion(str, Enum):
    CONTENT = "content"
    STRUCTURE = "structure" 
    CLARITY = "clarity"
    DELIVERY = "delivery"

class ScoreLevel(BaseModel):
    score: int = Field(..., ge=1, le=5)
    description: str
    indicators: List[str]

class CriterionScore(BaseModel):
    criterion: RubricCriterion
    score: int = Field(..., ge=1, le=5)
    feedback: str
    transcript_references: List[Dict] = Field(default_factory=list)
    
class ActionableTip(BaseModel):
    title: str
    description: str
    transcript_lines: List[str] = Field(default_factory=list)
    timestamp_ranges: List[Dict] = Field(default_factory=list)
    priority: int = Field(..., ge=1, le=3)

class InterviewEvaluation(BaseModel):
    scores: List[CriterionScore]
    overall_score: float
    actionable_tips: List[ActionableTip]
    transcript_highlights: List[Dict] = Field(default_factory=list)
    duration_analysis: Dict = Field(default_factory=dict)
    # Newly added enriched feedback fields
    strengths: List[str] = Field(default_factory=list)
    areas_for_improvement: List[str] = Field(default_factory=list)
    vocal_feedback: List[str] = Field(default_factory=list)  # Up to 5 vocal coaching points
    follow_up_question: Optional[str] = None  # If answer unsatisfactory, prompt user to retry focusing on weaknesses

class InterviewRubric(BaseModel):
    criteria: Dict[RubricCriterion, List[ScoreLevel]]
    max_duration: int = 60  # seconds
    
    @classmethod
    def get_default_rubric(cls):
        return cls(
            criteria={
                RubricCriterion.CONTENT: [
                    ScoreLevel(score=1, description="Minimal understanding", indicators=["Incorrect facts", "Missing key concepts"]),
                    ScoreLevel(score=2, description="Basic understanding", indicators=["Some correct facts", "Limited depth"]),
                    ScoreLevel(score=3, description="Good understanding", indicators=["Most facts correct", "Adequate depth"]),
                    ScoreLevel(score=4, description="Strong understanding", indicators=["All facts correct", "Good depth", "Some insights"]),
                    ScoreLevel(score=5, description="Excellent understanding", indicators=["Perfect accuracy", "Deep insights", "Advanced concepts"])
                ],
                RubricCriterion.STRUCTURE: [
                    ScoreLevel(score=1, description="No clear structure", indicators=["Random order", "No logical flow"]),
                    ScoreLevel(score=2, description="Minimal structure", indicators=["Some organization", "Unclear transitions"]),
                    ScoreLevel(score=3, description="Clear structure", indicators=["Logical flow", "Clear sections"]),
                    ScoreLevel(score=4, description="Well-structured", indicators=["Strong organization", "Smooth transitions"]),
                    ScoreLevel(score=5, description="Perfectly structured", indicators=["Compelling narrative", "Seamless flow"])
                ],
                RubricCriterion.CLARITY: [
                    ScoreLevel(score=1, description="Very unclear", indicators=["Confusing language", "Hard to follow"]),
                    ScoreLevel(score=2, description="Somewhat unclear", indicators=["Some confusing parts", "Occasional clarity"]),
                    ScoreLevel(score=3, description="Clear", indicators=["Generally understandable", "Good word choice"]),
                    ScoreLevel(score=4, description="Very clear", indicators=["Easy to follow", "Precise language"]),
                    ScoreLevel(score=5, description="Crystal clear", indicators=["Perfect clarity", "Engaging language"])
                ],
                RubricCriterion.DELIVERY: [
                    ScoreLevel(score=1, description="Poor delivery", indicators=["Many fillers", "Monotone", "Poor pacing"]),
                    ScoreLevel(score=2, description="Weak delivery", indicators=["Some fillers", "Limited variation"]),
                    ScoreLevel(score=3, description="Good delivery", indicators=["Few fillers", "Good pace", "Some variation"]),
                    ScoreLevel(score=4, description="Strong delivery", indicators=["Confident tone", "Good rhythm", "Engaging"]),
                    ScoreLevel(score=5, description="Excellent delivery", indicators=["Very confident", "Perfect pacing", "Compelling"])
                ]
            }
        )
    