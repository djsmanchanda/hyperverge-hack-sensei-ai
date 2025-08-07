from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import tempfile
import os
from ..services.interview_evaluator import InterviewEvaluator
from ..models.interview_rubric import InterviewEvaluation

router = APIRouter(prefix="/interview", tags=["interview"])

class InterviewRequest(BaseModel):
    question: str
    context: Optional[dict] = None
    max_duration: Optional[int] = 60

class InterviewResponse(BaseModel):
    evaluation: InterviewEvaluation
    transcript: str
    audio_duration: float

@router.post("/evaluate-audio", response_model=InterviewResponse)
async def evaluate_audio_interview(
    audio_file: UploadFile = File(...),
    question: str = "",
    context: Optional[str] = None
):
    """Evaluate an audio interview response with transcription and rubric scoring"""
    
    if not audio_file:
        raise HTTPException(status_code=400, detail="Audio file is required")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        content = await audio_file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
    
    try:
        # Initialize evaluator
        evaluator = InterviewEvaluator()
        
        # Parse context if provided
        context_dict = json.loads(context) if context else {}
        
        # Evaluate the audio response
        evaluation = await evaluator.evaluate_audio_response(
            audio_file_path=temp_file_path,
            question=question,
            context=context_dict
        )
        
        # Get transcript for response
        transcription = evaluator.transcription_service.transcribe_with_timestamps(temp_file_path)
        
        return InterviewResponse(
            evaluation=evaluation,
            transcript=transcription["text"],
            audio_duration=evaluation.duration_analysis["total_duration"]
        )
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@router.get("/prompts/{category}")
async def get_interview_prompts(category: str):
    """Get curated interview prompts by category"""
    
    prompts = {
        "cs": [
            "Explain the LRU cache algorithm in 60 seconds or less.",
            "Describe how you would design a URL shortener like bit.ly.",
            "Walk me through the process of how a web browser loads a webpage.",
            "Explain the difference between SQL and NoSQL databases.",
            "Describe what happens when you type a URL into your browser.",
            # ... more CS prompts
        ],
        "hr": [
            "Tell me about a time you had to work with a difficult team member.",
            "Describe your greatest professional achievement.",
            "Why are you interested in this role?",
            "How do you handle stress and pressure?",
            "Where do you see yourself in 5 years?",
            # ... more HR prompts
        ]
    }
    
    if category.lower() not in prompts:
        raise HTTPException(status_code=404, detail="Category not found")
    
    return {"prompts": prompts[category.lower()]}