import whisper
import json
from typing import Dict, List, Tuple
import re

class TranscriptionService:
    def __init__(self):
        self.model = whisper.load_model("base")
    
    def transcribe_with_timestamps(self, audio_file_path: str) -> Dict:
        """Transcribe audio with word-level timestamps"""
        result = self.model.transcribe(
            audio_file_path, 
            word_timestamps=True,
            verbose=True
        )
        
        return {
            "text": result["text"],
            "segments": result["segments"],
            "words": self._extract_words_with_timestamps(result["segments"])
        }
    
    def _extract_words_with_timestamps(self, segments: List) -> List[Dict]:
        """Extract individual words with timestamps"""
        words = []
        for segment in segments:
            if "words" in segment:
                for word in segment["words"]:
                    words.append({
                        "word": word["word"],
                        "start": word["start"],
                        "end": word["end"],
                        "confidence": word.get("probability", 0.0)
                    })
        return words
    
    def detect_fillers(self, words: List[Dict]) -> List[Dict]:
        """Detect filler words and phrases"""
        filler_patterns = [
            r'\b(um|uh|er|ah)\b',
            r'\b(like|you know|basically|actually|literally)\b',
            r'\b(so|well|right)\b(?=\s)',
        ]
        
        fillers = []
        for word in words:
            for pattern in filler_patterns:
                if re.search(pattern, word["word"].lower()):
                    fillers.append({
                        **word,
                        "type": "filler",
                        "pattern": pattern
                    })
        return fillers