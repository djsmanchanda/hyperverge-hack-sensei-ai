import base64
import tempfile
import os
from typing import Dict, List
import re
import json
from openai import OpenAI


def prepare_audio_input_for_ai(audio_data: bytes):
    return base64.b64encode(audio_data).decode("utf-8")


class AudioTranscriptionService:
    def __init__(self):
        self.client = None
        try:
            from api.settings import settings

            self.client = OpenAI(api_key=settings.openai_api_key)
            print("✅ OpenAI transcription client initialized successfully")
        except ImportError as e:
            print(f"❌ OpenAI not available: {e}")
        except Exception as e:
            print(f"❌ Error initializing OpenAI client: {e}")

    def transcribe_with_analysis(self, audio_data: bytes) -> Dict:
        """Transcribe audio using OpenAI API and analyze speech patterns"""
        if not self.client:
            return {"error": "OpenAI client not available"}

        # Save audio data to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name

        try:
            # Transcribe using OpenAI API
            with open(temp_file_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-1",  # Use whisper-1 instead of gpt-4o-transcribe for now
                    file=audio_file,
                    response_format="verbose_json",  # Get timestamps
                    timestamp_granularities=["word"],  # Word-level timestamps
                )

            # Convert OpenAI response to our expected format
            result = {
                "text": transcription.text,
                "segments": getattr(transcription, "segments", []),
                "words": getattr(transcription, "words", []),
            }

            # Analyze speech patterns
            analysis = self._analyze_speech_patterns(result)

            return {
                "transcript": result["text"],
                "segments": result.get("segments", []),
                "analysis": analysis,
                "duration": self._get_total_duration(result),
            }

        except Exception as e:
            print(f"Error during transcription: {e}")
            return {"error": str(e)}
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def _get_total_duration(self, result: Dict) -> float:
        """Extract total duration from transcription result"""
        if result.get("segments"):
            return result["segments"][-1].get("end", 0)
        elif result.get("words"):
            return result["words"][-1].get("end", 0)
        return 0

    def _analyze_speech_patterns(self, transcription_result: Dict) -> Dict:
        """Analyze speech patterns for feedback"""
        segments = transcription_result.get("segments", [])
        words = transcription_result.get("words", [])
        text = transcription_result.get("text", "")

        # If no word-level timestamps, extract from segments
        if not words and segments:
            words = []
            for segment in segments:
                if "words" in segment and segment["words"]:
                    words.extend(segment["words"])
                else:
                    # Fallback: create word entries from segment
                    segment_words = segment.get("text", "").split()
                    segment_duration = segment.get("end", 0) - segment.get("start", 0)
                    word_duration = segment_duration / len(segment_words) if segment_words else 0

                    for i, word in enumerate(segment_words):
                        words.append(
                            {
                                "word": word,
                                "start": segment.get("start", 0) + (i * word_duration),
                                "end": segment.get("start", 0) + ((i + 1) * word_duration),
                            }
                        )

        # Detect filler words
        filler_patterns = [
            r"\b(um|uh|er|ah|hmm)\b",
            r"\b(like|you know|basically|actually|literally)\b",
            r"\b(so|well|right|okay)\b(?=\s)",
        ]

        fillers = []
        filler_count = 0

        for word_info in words:
            word = word_info.get("word", "").lower().strip()
            for pattern in filler_patterns:
                if re.search(pattern, word):
                    fillers.append(
                        {
                            "word": word,
                            "start": word_info.get("start", 0),
                            "end": word_info.get("end", 0),
                        }
                    )
                    filler_count += 1
                    break

        # Calculate speaking rate
        total_duration = self._get_total_duration(transcription_result)
        word_count = len(words)
        wpm = (word_count / total_duration * 60) if total_duration > 0 else 0

        # Detect long pauses
        long_pauses = []
        if segments:
            for i in range(1, len(segments)):
                gap = segments[i]["start"] - segments[i - 1]["end"]
                if gap > 2.0:  # Pause longer than 2 seconds
                    long_pauses.append(
                        {
                            "start": segments[i - 1]["end"],
                            "end": segments[i]["start"],
                            "duration": gap,
                        }
                    )

        return {
            "word_count": word_count,
            "filler_count": filler_count,
            "fillers": fillers,
            "speaking_rate_wpm": wpm,
            "long_pauses": long_pauses,
            "total_duration": total_duration,
        }


# Global instance
audio_service = AudioTranscriptionService()
