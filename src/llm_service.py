import os
import json
from typing import Dict, Any
from openai import AsyncOpenAI
from pydantic import BaseModel

class LLMService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def get_structured_response(self, system_prompt: str, user_message: str, response_format: Dict[str, Any]):
        # For now we request JSON output with simple schema instructions
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        response = await self.client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            response_format={"type": "json_schema", "json_schema": {"name": "resp_schema", "schema": {"type": "object", "properties": response_format, "additionalProperties": True}}},
            temperature=0
        )
        content = response.choices[0].message.content
        return json.loads(content) if content else {}
