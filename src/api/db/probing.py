import uuid
from typing import List, Dict, Optional
from api.utils.db import get_new_db_connection
import json

async def create_probing_session(
    user_id: str,
    question_id: str, 
    task_id: str,
    initial_correct_answer: str
) -> str:
    """Create a new probing session"""
    session_uuid = str(uuid.uuid4())
    
    async with get_new_db_connection() as conn:
        await conn.execute(
            """
            INSERT INTO probing_sessions 
            (user_id, question_id, task_id, session_uuid, initial_correct_answer, session_state)
            VALUES (?, ?, ?, ?, ?, 'correct_answer')
            """,
            (user_id, question_id, task_id, session_uuid, initial_correct_answer),
        )
        await conn.commit()
    
    return session_uuid

async def update_probing_session_with_question(
    session_uuid: str,
    probing_question: str,
    probing_type: str
):
    """Update session with probing question"""
    async with get_new_db_connection() as conn:
        await conn.execute(
            """
            UPDATE probing_sessions 
            SET probing_question = ?, probing_type = ?, session_state = 'probing'
            WHERE session_uuid = ?
            """,
            (probing_question, probing_type, session_uuid),
        )
        await conn.commit()

async def update_probing_session_with_response(
    session_uuid: str,
    student_response: str,
    understanding_demonstrated: bool,
    certification_achieved: bool = False,
    mastery_level: str = None,
    concepts_mastered: List[str] = None
):
    """Update session with student response and evaluation"""
    async with get_new_db_connection() as conn:
        await conn.execute(
            """
            UPDATE probing_sessions 
            SET student_probing_response = ?, 
                understanding_demonstrated = ?,
                certification_achieved = ?,
                mastery_level = ?,
                concepts_mastered = ?,
                completed_at = CURRENT_TIMESTAMP,
                session_state = CASE WHEN ? THEN 'certified' ELSE 'probing' END
            WHERE session_uuid = ?
            """,
            (
                student_response,
                int(bool(understanding_demonstrated)),
                int(bool(certification_achieved)),
                mastery_level,
                json.dumps(concepts_mastered or []),
                int(bool(certification_achieved)),
                session_uuid,
            ),
        )
        await conn.commit()

async def create_understanding_certification(
    user_id: str,
    question_id: str,
    task_id: str,
    session_uuid: str,
    mastery_level: str,
    concepts_mastered: List[str],
    probing_attempts: int = 1,
    total_session_time: int = None
):
    """Record understanding certification"""
    async with get_new_db_connection() as conn:
        await conn.execute(
            """
            INSERT INTO understanding_certifications 
            (user_id, question_id, task_id, session_uuid, mastery_level, 
             concepts_mastered, probing_attempts, total_session_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                question_id,
                task_id,
                session_uuid,
                mastery_level,
                json.dumps(concepts_mastered or []),
                probing_attempts,
                total_session_time,
            ),
        )
        await conn.commit()

async def get_user_certifications(user_id: str, task_id: str = None):
    """Get user's understanding certifications"""
    async with get_new_db_connection() as conn:
        if task_id:
            cursor = await conn.execute(
                """
                SELECT * FROM understanding_certifications 
                WHERE user_id = ? AND task_id = ?
                ORDER BY certified_at DESC
                """,
                (user_id, task_id),
            )
        else:
            cursor = await conn.execute(
                """
                SELECT * FROM understanding_certifications 
                WHERE user_id = ?
                ORDER BY certified_at DESC
                """,
                (user_id,),
            )
        rows = await cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in rows]

async def get_probing_session(session_uuid: str):
    """Get probing session details"""
    async with get_new_db_connection() as conn:
        cursor = await conn.execute(
            """
            SELECT * FROM probing_sessions WHERE session_uuid = ?
            """,
            (session_uuid,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        columns = [col[0] for col in cursor.description]
        return dict(zip(columns, row))