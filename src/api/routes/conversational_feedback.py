from fastapi import APIRouter, HTTPException, Form, UploadFile, File
from typing import Dict, List
import json
import asyncio
from datetime import datetime
from pydantic import BaseModel

from api.db.task import get_task
from api.models import (
    TaskType, 
    ConversationalFeedbackResponse,
    ConversationalFeedbackSubmission
)
from api.llm import run_llm_with_instructor
from api.settings import settings
from api.utils.audio import prepare_audio_input_for_ai
from api.utils.s3 import upload_file_to_s3, get_media_upload_s3_key_from_uuid
import uuid
import os

router = APIRouter()


class AudioSubmissionRequest(BaseModel):
    user_id: int
    task_id: int
    audio_file_uuid: str


class FeedbackScoreItem(BaseModel):
    criterion: str
    score: int
    feedback: str
    line_references: List[str] = []


class ConversationalFeedbackResult(BaseModel):
    transcription: str
    overall_feedback: str
    scores: List[FeedbackScoreItem]
    overall_score: float
    filler_count: int
    duration_seconds: float


@router.post("/submit-audio", response_model=ConversationalFeedbackResponse)
async def submit_audio_for_transcription_and_feedback(
    user_id: int = Form(),
    task_id: int = Form(),
    audio_file: UploadFile = File(...)
) -> ConversationalFeedbackResponse:
    """
    Submit audio for conversational feedback analysis.
    
    This endpoint:
    1. Receives audio file upload
    2. Stores it with UUID
    3. Transcribes using OpenAI Whisper
    4. Analyzes against rubric criteria
    5. Returns structured feedback
    """
    try:
        # Generate UUID for the audio file
        audio_uuid = str(uuid.uuid4())
        
        # Read audio file content
        audio_content = await audio_file.read()
        
        # Store audio file (S3 or local)
        if settings.s3_folder_name:
            s3_key = get_media_upload_s3_key_from_uuid(audio_uuid, "wav")
            upload_file_to_s3(audio_content, s3_key)
        else:
            # Save locally
            os.makedirs(settings.local_upload_folder, exist_ok=True)
            file_path = os.path.join(settings.local_upload_folder, f"{audio_uuid}.wav")
            with open(file_path, "wb") as f:
                f.write(audio_content)
        
        # Get task details to retrieve rubric and configuration
        task = await get_task(task_id)
        if not task or task["type"] != TaskType.CONVERSATIONAL_FEEDBACK:
            raise HTTPException(status_code=404, detail="Conversational feedback task not found")
        
        config = task.get("config", {})
        rubric = config.get("rubric", {})
        prompt_blocks = config.get("prompt", [])
        max_duration = config.get("maxDuration", 120)
        
        # Extract prompt text from blocks
        prompt_text = ""
        for block in prompt_blocks:
            if block.get("type") == "paragraph" and "content" in block:
                for content_item in block["content"]:
                    if content_item.get("type") == "text":
                        prompt_text += content_item.get("text", "")
        
        # Step 1: Transcribe audio using OpenAI Whisper
        transcription = await transcribe_audio_with_whisper(audio_content)
        
        # Step 2: Analyze transcription against rubric
        feedback_result = await analyze_conversational_feedback(
            transcription=transcription,
            prompt_text=prompt_text,
            rubric=rubric,
            max_duration=max_duration
        )
        
        # Create response
        response = ConversationalFeedbackResponse(
            transcription=feedback_result.transcription,
            feedback={
                "overall": feedback_result.overall_feedback,
                "criteria": [
                    {
                        "name": score.criterion,
                        "score": score.score,
                        "feedback": score.feedback,
                        "line_references": score.line_references
                    }
                    for score in feedback_result.scores
                ],
                "filler_count": feedback_result.filler_count,
                "duration_seconds": feedback_result.duration_seconds
            },
            scores=[
                {
                    "criterion": score.criterion,
                    "score": score.score,
                    "max_score": next(
                        (c["maxScore"] for c in rubric.get("criteria", []) if c["name"] == score.criterion),
                        10
                    )
                }
                for score in feedback_result.scores
            ],
            overall_score=feedback_result.overall_score
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


async def transcribe_audio_with_whisper(audio_content: bytes) -> str:
    """Transcribe audio using OpenAI Whisper API."""
    import openai
    
    try:
        client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        
        # Create a temporary file-like object for the API
        from io import BytesIO
        audio_file = BytesIO(audio_content)
        audio_file.name = "audio.wav"
        
        # Call Whisper API
        transcript = await client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
        
        return transcript
        
    except Exception as e:
        raise Exception(f"Transcription failed: {str(e)}")


async def analyze_conversational_feedback(
    transcription: str,
    prompt_text: str,
    rubric: Dict,
    max_duration: int
) -> ConversationalFeedbackResult:
    """
    Analyze conversational feedback using AI to score against rubric criteria.
    """
    
    # Extract rubric criteria
    criteria = rubric.get("criteria", [])
    if not criteria:
        raise Exception("No rubric criteria found")
    
    # Create scoring model based on rubric
    class ConversationalAnalysis(BaseModel):
        transcription_analysis: str
        overall_feedback: str
        criterion_scores: List[Dict]
        overall_score: float
        filler_count: int
        duration_assessment: str
    
    # Build system prompt for analysis
    criteria_descriptions = []
    for criterion in criteria:
        criteria_descriptions.append(
            f"- **{criterion['name']}** (max {criterion['maxScore']} points): {criterion['description']}"
        )
    
    criteria_text = "\n".join(criteria_descriptions)
    
    system_prompt = f"""You are an expert communication coach analyzing a recorded response to this prompt:

**Original Prompt:** {prompt_text}

**Transcription to Analyze:** {transcription}

**Scoring Rubric:**
{criteria_text}

**Analysis Requirements:**
1. Provide detailed feedback for each criterion
2. Score each criterion based on the rubric (0 to max points)
3. Count filler words ("um", "uh", "like", "you know", etc.)
4. Assess if response fits time constraints (max {max_duration} seconds)
5. Provide actionable feedback tied to specific parts of the transcription
6. Calculate overall score as average of criterion scores

**Output Format:**
- transcription_analysis: Brief analysis of content quality
- overall_feedback: Summary of strengths and improvement areas
- criterion_scores: Array of objects with "name", "score", "max_score", "feedback", "line_references"
- overall_score: Float (0-10 scale)
- filler_count: Integer count of filler words
- duration_assessment: Brief note on timing/pacing"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Please analyze this transcription according to the rubric: {transcription}"}
    ]
    
    try:
        # Use AI to analyze the transcription
        analysis = await run_llm_with_instructor(
            api_key=settings.openai_api_key,
            model="gpt-4o",
            messages=messages,
            response_model=ConversationalAnalysis,
            max_completion_tokens=2048
        )
        
        # Convert to our result format
        scores = []
        for criterion_score in analysis.criterion_scores:
            scores.append(FeedbackScoreItem(
                criterion=criterion_score.get("name", ""),
                score=criterion_score.get("score", 0),
                feedback=criterion_score.get("feedback", ""),
                line_references=criterion_score.get("line_references", [])
            ))
        
        return ConversationalFeedbackResult(
            transcription=transcription,
            overall_feedback=analysis.overall_feedback,
            scores=scores,
            overall_score=analysis.overall_score,
            filler_count=analysis.filler_count,
            duration_seconds=0.0  # TODO: Calculate from audio file
        )
        
    except Exception as e:
        raise Exception(f"Analysis failed: {str(e)}")


@router.get("/submission/{submission_id}")
async def get_conversational_feedback_submission(submission_id: int):
    """Retrieve a conversational feedback submission by ID."""
    # TODO: Implement database retrieval
    raise HTTPException(status_code=501, detail="Not implemented yet")


@router.get("/task/{task_id}/submissions")
async def get_task_submissions(task_id: int) -> List[ConversationalFeedbackSubmission]:
    """Get all submissions for a conversational feedback task."""
    # TODO: Implement database retrieval
    raise HTTPException(status_code=501, detail="Not implemented yet")
