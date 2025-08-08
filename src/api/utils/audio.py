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

        # Detect language switching
        language_analysis = self._detect_language_switching(text, segments)
        
        # Analyze coherence and logical flow
        coherence_analysis = self._analyze_coherence(text, segments)
        
        # Detect repetitive content
        repetition_analysis = self._analyze_repetition(text, words)
        
        # Calculate filler word density per segment
        filler_density = self._calculate_filler_density(fillers, segments)

        return {
            "word_count": word_count,
            "filler_count": filler_count,
            "fillers": fillers,
            "speaking_rate_wpm": wpm,
            "long_pauses": long_pauses,
            "total_duration": total_duration,
            "language_analysis": language_analysis,
            "coherence_analysis": coherence_analysis,
            "repetition_analysis": repetition_analysis,
            "overall_clarity_score": self._calculate_clarity_score(
                filler_count, word_count, len(long_pauses), coherence_analysis, language_analysis, repetition_analysis
            )
        }

    def _classify_filler_type(self, word: str) -> str:
        """Classify the type of filler word"""
        hesitation_sounds = ["um", "uh", "er", "ah", "hmm", "mmm", "uhm"]
        discourse_markers = ["like", "you know", "basically", "actually", "literally", "obviously", "clearly"]
        transition_fillers = ["so", "well", "right", "okay", "now", "then", "anyway", "however"]
        phrase_fillers = ["i mean", "sort of", "kind of", "you see", "you understand", "if you will"]
        vague_endings = ["and stuff", "or whatever", "or something", "et cetera", "etc"]
        
        word_lower = word.lower().strip()
        if word_lower in hesitation_sounds:
            return "hesitation"
        elif word_lower in discourse_markers:
            return "discourse_marker"
        elif word_lower in transition_fillers:
            return "transition"
        elif word_lower in phrase_fillers:
            return "phrase_filler"
        elif word_lower in vague_endings:
            return "vague_ending"
        else:
            return "other"

    def _detect_language_switching(self, text: str, segments: List) -> Dict:
        """Detect if the speaker is switching between languages"""
        # Common patterns that indicate language switching
        language_indicators = {
            'spanish': ['que', 'pero', 'y', 'en', 'el', 'la', 'es', 'no', 'si', 'con', 'un', 'una', 'por', 'para', 'como', 'muy', 'bien', 'más', 'también', 'gracias', 'hola', 'adiós'],
            'french': ['que', 'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour', 'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se', 'pas', 'tout', 'plus', 'par', 'pouvoir', 'mais', 'oui', 'bonjour', 'merci', 'au revoir'],
            'german': ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich', 'des', 'auf', 'für', 'ist', 'im', 'dem', 'nicht', 'ein', 'eine', 'als', 'auch', 'es', 'an', 'werden', 'aus', 'er', 'hat', 'dass', 'sie', 'nach', 'wird', 'bei', 'einer', 'um', 'am', 'sind', 'noch', 'wie', 'einem', 'über', 'einen', 'so', 'zum', 'war', 'haben', 'nur', 'oder', 'aber', 'vor', 'zur', 'bis', 'mehr', 'durch', 'man', 'sein', 'wurde', 'sei', 'in', 'hallo', 'danke', 'auf wiedersehen'],
            'hindi': ['है', 'में', 'की', 'को', 'का', 'के', 'से', 'पर', 'और', 'यह', 'वह', 'एक', 'हमारे', 'तुम', 'आप', 'मैं', 'हम', 'था', 'थी', 'करना', 'होना', 'जाना', 'कहना', 'देना', 'लेना', 'आना', 'जो', 'भी', 'नहीं', 'तो', 'ही', 'अच्छा', 'बुरा', 'नमस्ते', 'धन्यवाद'],
            'arabic': ['في', 'من', 'إلى', 'على', 'أن', 'هذا', 'هذه', 'التي', 'الذي', 'كان', 'لا', 'ما', 'قد', 'له', 'أو', 'عن', 'كما', 'بعد', 'قبل', 'عند', 'حتى', 'بين', 'تحت', 'فوق', 'أمام', 'خلف', 'يمين', 'يسار', 'السلام عليكم', 'شكرا', 'مع السلامة']
        }
        
        words = text.lower().split()
        detected_languages = set()
        language_switches = []
        
        # Count words in each language
        language_counts = {lang: 0 for lang in language_indicators.keys()}
        
        for word in words:
            for lang, indicators in language_indicators.items():
                if word in indicators:
                    language_counts[lang] += 1
                    detected_languages.add(lang)
        
        # Detect significant language switching (more than 2 words in a non-English language)
        significant_languages = [lang for lang, count in language_counts.items() if count >= 2]
        
        is_multilingual = len(significant_languages) > 0
        switching_frequency = len(significant_languages)
        
        return {
            "is_multilingual": is_multilingual,
            "detected_languages": list(detected_languages),
            "significant_languages": significant_languages,
            "language_counts": language_counts,
            "switching_frequency": switching_frequency,
            "multilingual_score": min(10, max(1, 10 - switching_frequency * 2))  # 1-10 scale
        }

    def _analyze_coherence(self, text: str, segments: List) -> Dict:
        """Analyze logical organization and clarity of ideas"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Coherence indicators
        transition_words = [
            'first', 'second', 'third', 'finally', 'moreover', 'furthermore', 'however', 
            'therefore', 'consequently', 'in conclusion', 'to summarize', 'on the other hand',
            'for example', 'such as', 'in addition', 'meanwhile', 'nevertheless', 'thus',
            'accordingly', 'similarly', 'likewise', 'in contrast', 'as a result'
        ]
        
        logical_connectors = [
            'because', 'since', 'although', 'while', 'whereas', 'if', 'unless', 'until',
            'before', 'after', 'when', 'where', 'why', 'how', 'what', 'which', 'that'
        ]
        
        # Count coherence markers
        transition_count = 0
        logical_connector_count = 0
        
        words = text.lower().split()
        for word in words:
            if word in transition_words:
                transition_count += 1
            if word in logical_connectors:
                logical_connector_count += 1
        
        # Analyze sentence structure variety
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        sentence_variety = len(set(sentence_lengths)) / len(sentence_lengths) if sentence_lengths else 0
        
        # Check for abrupt topic changes (simple heuristic)
        topic_consistency = self._analyze_topic_consistency(sentences)
        
        # Calculate coherence score (1-10)
        coherence_factors = [
            min(10, transition_count * 2),  # Transition words boost score
            min(10, logical_connector_count),  # Logical connectors boost score
            min(10, max(1, avg_sentence_length / 2)),  # Reasonable sentence length
            min(10, sentence_variety * 10),  # Sentence variety
            topic_consistency * 10  # Topic consistency
        ]
        
        coherence_score = sum(coherence_factors) / len(coherence_factors)
        
        return {
            "coherence_score": round(min(10, max(1, coherence_score)), 1),
            "transition_words_count": transition_count,
            "logical_connectors_count": logical_connector_count,
            "average_sentence_length": round(avg_sentence_length, 1),
            "sentence_count": len(sentences),
            "sentence_variety_score": round(sentence_variety, 2),
            "topic_consistency_score": round(topic_consistency, 2),
            "coherence_issues": self._identify_coherence_issues(sentences, transition_count, logical_connector_count)
        }

    def _analyze_topic_consistency(self, sentences: List[str]) -> float:
        """Analyze how consistent the topic is throughout the speech"""
        if len(sentences) < 2:
            return 1.0
        
        # Simple keyword overlap between consecutive sentences
        consistency_scores = []
        
        for i in range(len(sentences) - 1):
            words1 = set(sentences[i].lower().split())
            words2 = set(sentences[i + 1].lower().split())
            
            # Remove common stop words for better analysis
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'this', 'that', 'these', 'those'}
            
            words1 = words1 - stop_words
            words2 = words2 - stop_words
            
            if len(words1) == 0 or len(words2) == 0:
                consistency_scores.append(0.5)
            else:
                overlap = len(words1.intersection(words2))
                union = len(words1.union(words2))
                consistency_scores.append(overlap / union if union > 0 else 0)
        
        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 1.0

    def _identify_coherence_issues(self, sentences: List[str], transition_count: int, logical_connector_count: int) -> List[str]:
        """Identify specific coherence issues"""
        issues = []
        
        if len(sentences) > 3 and transition_count == 0:
            issues.append("Lacks transition words to connect ideas")
        
        if len(sentences) > 2 and logical_connector_count == 0:
            issues.append("Missing logical connectors to show relationships between ideas")
        
        sentence_lengths = [len(s.split()) for s in sentences]
        if sentence_lengths:
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            if avg_length < 3:
                issues.append("Sentences are too short and choppy")
            elif avg_length > 30:
                issues.append("Sentences are too long and complex")
        
        if len(set(sentence_lengths)) == 1 and len(sentences) > 2:
            issues.append("All sentences have similar length - add variety")
        
        return issues

    def _analyze_repetition(self, text: str, words: List) -> Dict:
        """Analyze repetitive content and redundancy"""
        word_list = [self._get_word_property(w, "word", "").lower().strip() for w in words if self._get_word_property(w, "word")]
        
        # Remove common stop words for meaningful repetition analysis
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can'}
        
        meaningful_words = [w for w in word_list if w not in stop_words and len(w) > 2]
        
        # Count word frequencies
        word_freq = {}
        for word in meaningful_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Find repetitive words (appearing more than expected)
        total_meaningful_words = len(meaningful_words)
        repetitive_words = []
        excessive_repetitions = []
        
        for word, count in word_freq.items():
            expected_frequency = 1 / len(set(meaningful_words)) if meaningful_words else 0
            actual_frequency = count / total_meaningful_words if total_meaningful_words > 0 else 0
            
            if count > 2 and actual_frequency > expected_frequency * 3:  # More than 3x expected
                repetitive_words.append({"word": word, "count": count, "frequency": actual_frequency})
            
            if count > 5:  # Excessive repetition
                excessive_repetitions.append({"word": word, "count": count})
        
        # Analyze phrase repetition
        phrases = self._extract_phrases(text)
        repeated_phrases = self._find_repeated_phrases(phrases)
        
        # Calculate repetition score (1-10, where 10 is no repetition)
        repetition_penalty = len(repetitive_words) + len(excessive_repetitions) * 2 + len(repeated_phrases) * 3
        repetition_score = max(1, 10 - repetition_penalty)
        
        return {
            "repetition_score": repetition_score,
            "repetitive_words": repetitive_words,
            "excessive_repetitions": excessive_repetitions,
            "repeated_phrases": repeated_phrases,
            "unique_word_ratio": len(set(meaningful_words)) / len(meaningful_words) if meaningful_words else 1.0,
            "repetition_issues": self._identify_repetition_issues(repetitive_words, excessive_repetitions, repeated_phrases)
        }

    def _extract_phrases(self, text: str) -> List[str]:
        """Extract 2-4 word phrases from text"""
        words = text.lower().split()
        phrases = []
        
        for length in [2, 3, 4]:
            for i in range(len(words) - length + 1):
                phrase = " ".join(words[i:i+length])
                phrases.append(phrase)
        
        return phrases

    def _find_repeated_phrases(self, phrases: List[str]) -> List[Dict]:
        """Find phrases that are repeated"""
        phrase_freq = {}
        for phrase in phrases:
            phrase_freq[phrase] = phrase_freq.get(phrase, 0) + 1
        
        repeated = []
        for phrase, count in phrase_freq.items():
            if count > 1 and len(phrase.split()) >= 2:  # At least 2 words, repeated
                repeated.append({"phrase": phrase, "count": count})
        
        return sorted(repeated, key=lambda x: x["count"], reverse=True)[:5]  # Top 5

    def _identify_repetition_issues(self, repetitive_words: List, excessive_repetitions: List, repeated_phrases: List) -> List[str]:
        """Identify specific repetition issues"""
        issues = []
        
        if excessive_repetitions:
            most_repeated = max(excessive_repetitions, key=lambda x: x["count"])
            issues.append(f"Excessive repetition of '{most_repeated['word']}' ({most_repeated['count']} times)")
        
        if len(repetitive_words) > 3:
            issues.append(f"Too many repetitive words ({len(repetitive_words)} different words overused)")
        
        if repeated_phrases:
            issues.append(f"Repeated phrases detected: '{repeated_phrases[0]['phrase']}' used {repeated_phrases[0]['count']} times")
        
        if len(issues) == 0 and (repetitive_words or repeated_phrases):
            issues.append("Some word/phrase repetition detected - consider using synonyms")
        
        return issues

    def _calculate_filler_density(self, fillers: List, segments: List) -> Dict:
        """Calculate filler word density per segment"""
        if not segments:
            return {"overall_density": 0, "segment_densities": []}
        
        segment_densities = []
        for segment in segments:
            segment_start = segment.get("start", 0)
            segment_end = segment.get("end", 0)
            segment_text = segment.get("text", "")
            segment_words = len(segment_text.split())
            
            # Count fillers in this segment
            segment_fillers = [f for f in fillers if segment_start <= f["start"] <= segment_end]
            density = len(segment_fillers) / segment_words if segment_words > 0 else 0
            
            segment_densities.append({
                "start": segment_start,
                "end": segment_end,
                "filler_count": len(segment_fillers),
                "word_count": segment_words,
                "density": density
            })
        
        overall_density = len(fillers) / sum(s["word_count"] for s in segment_densities) if segment_densities else 0
        
        return {
            "overall_density": overall_density,
            "segment_densities": segment_densities,
            "high_density_segments": [s for s in segment_densities if s["density"] > 0.2]  # More than 20% fillers
        }

    def _calculate_clarity_score(self, filler_count: int, word_count: int, long_pause_count: int, 
                               coherence_analysis: Dict, language_analysis: Dict, repetition_analysis: Dict) -> float:
        """Calculate overall clarity score from 1-10"""
        
        # Filler score (1-10, where fewer fillers = higher score)
        filler_ratio = filler_count / word_count if word_count > 0 else 0
        filler_score = max(1, 10 - (filler_ratio * 30))  # Penalty for high filler ratio
        
        # Pause score (1-10, where fewer long pauses = higher score)
        pause_score = max(1, 10 - long_pause_count)
        
        # Get scores from other analyses
        coherence_score = coherence_analysis.get("coherence_score", 5)
        multilingual_score = language_analysis.get("multilingual_score", 10)
        repetition_score = repetition_analysis.get("repetition_score", 10)
        
        # Weight the different factors
        weighted_score = (
            filler_score * 0.25 +      # 25% weight on filler words
            pause_score * 0.15 +       # 15% weight on pauses
            coherence_score * 0.30 +   # 30% weight on coherence
            multilingual_score * 0.15 + # 15% weight on language consistency
            repetition_score * 0.15    # 15% weight on avoiding repetition
        )
        
        return round(min(10, max(1, weighted_score)), 1)


class SpeakerDiarizationService:
    """AssemblyAI-only speaker diarization (simple, predictable).

    If ASSEMBLYAI_API_KEY is absent or fails, returns single-speaker stub.
    """

    def __init__(self):
        print("✅ AssemblyAI diarization service initialized")

    @property
    def is_available(self) -> bool:
        try:
            from api.settings import settings
            return bool(settings.assemblyai_api_key)
        except Exception:
            return False

    def detect_speakers(self, audio_data: bytes, language_code: str = "en-US", min_speakers: int = 1, max_speakers: int = 5) -> Dict:
        from api.settings import settings
        if not settings.assemblyai_api_key:
            return {
                "num_speakers": 1,
                "speakers": ["SPEAKER_A"],
                "speaker_segments": [],
                "speaker_stats": {},
                "total_duration": 0,
                "is_multi_speaker": False,
                "confidence": None,
                "method": "assemblyai_disabled"
            }
        try:
            import requests, time
            headers = {"authorization": settings.assemblyai_api_key}
            up = requests.post(
                "https://api.assemblyai.com/v2/upload",
                headers=headers,
                data=audio_data,
                timeout=60,
            )
            up.raise_for_status()
            upload_url = up.json()["upload_url"]
            body = {
                "audio_url": upload_url,
                "speaker_labels": True,
                "speaker_options": {
                    "min_speakers_expected": min_speakers,
                    "max_speakers_expected": max_speakers,
                },
            }
            tr = requests.post("https://api.assemblyai.com/v2/transcript", json=body, headers=headers, timeout=30)
            tr.raise_for_status()
            tid = tr.json()["id"]
            for _ in range(30):
                res = requests.get(f"https://api.assemblyai.com/v2/transcript/{tid}", headers=headers, timeout=15).json()
                status = res.get("status")
                if status == "completed":
                    utt = res.get("utterances", [])
                    speakers = {u.get("speaker") for u in utt if u.get("speaker")}
                    segments = [
                        {
                            "speaker": f"SPEAKER_{u.get('speaker')}",
                            "start": u.get("start", 0) / 1000.0,
                            "end": u.get("end", 0) / 1000.0,
                            "duration": (u.get("end", 0) - u.get("start", 0)) / 1000.0,
                            "text": u.get("text", ""),
                        }
                        for u in utt
                    ]
                    stats = self._calculate_speaker_stats(
                        [{"speaker": s["speaker"], "duration": s["duration"]} for s in segments],
                        {seg["speaker"] for seg in segments},
                    )
                    return {
                        "num_speakers": len(speakers) or 1,
                        "speakers": sorted(list(speakers)) or ["SPEAKER_A"],
                        "speaker_segments": segments,
                        "speaker_stats": stats,
                        "total_duration": segments[-1]["end"] if segments else 0,
                        "is_multi_speaker": len(speakers) > 1,
                        "confidence": res.get("confidence"),
                        "method": "assemblyai",
                    }
                if status in {"error", "failed"}:
                    print(f"AssemblyAI diarization failed: {res}")
                    break
                time.sleep(2)
            print("⚠️ AssemblyAI diarization timeout; returning single-speaker stub")
        except Exception as e:
            print(f"⚠️ AssemblyAI diarization exception: {e}")
        return {
            "num_speakers": 1,
            "speakers": ["SPEAKER_A"],
            "speaker_segments": [],
            "speaker_stats": {},
            "total_duration": 0,
            "is_multi_speaker": False,
            "confidence": None,
            "method": "assemblyai_fallback"
        }
    
    def _fallback_speaker_detection(self, audio_data: bytes) -> Dict:
        """
        Fallback method when pyannote is not available
        Uses simple audio characteristics to estimate speakers
        """
        try:
            # Save to temporary file for analysis
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            # Simple analysis using soundfile if available
            try:
                import soundfile as sf
                data, sample_rate = sf.read(temp_file_path)
                duration = len(data) / sample_rate
                
                # Simple heuristics for speaker detection
                # Based on audio characteristics and duration
                estimated_speakers = self._estimate_speakers_from_audio(data, sample_rate)
                
                return {
                    "num_speakers": estimated_speakers,
                    "speakers": [f"Speaker_{i+1}" for i in range(estimated_speakers)],
                    "speaker_segments": [],
                    "speaker_stats": {},
                    "total_duration": duration,
                    "is_multi_speaker": estimated_speakers > 1,
                    "confidence": "low",  # Fallback has low confidence
                    "method": "fallback_estimation",
                    "warning": "Speaker diarization not available - using estimation"
                }
                
            except ImportError:
                # Most basic fallback - assume single speaker
                return {
                    "num_speakers": 1,
                    "speakers": ["Speaker_1"],
                    "speaker_segments": [],
                    "speaker_stats": {},
                    "total_duration": 60,  # Default estimate
                    "is_multi_speaker": False,
                    "confidence": "unknown",
                    "method": "basic_fallback",
                    "warning": "No audio analysis libraries available"
                }
        
        except Exception as e:
            print(f"Fallback speaker detection failed: {e}")
            return {
                "num_speakers": 1,
                "speakers": ["Speaker_1"],
                "speaker_segments": [],
                "speaker_stats": {},
                "total_duration": 0,
                "is_multi_speaker": False,
                "confidence": "unknown",
                "method": "error_fallback",
                "error": str(e)
            }
        finally:
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def _estimate_speakers_from_audio(self, audio_data, sample_rate) -> int:
        """
        Simple heuristic to estimate number of speakers from audio characteristics
        """
        try:
            import numpy as np
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Calculate basic features
            duration = len(audio_data) / sample_rate
            
            # Simple energy-based segmentation
            frame_size = int(0.5 * sample_rate)  # 0.5 second frames
            frames = []
            
            for i in range(0, len(audio_data) - frame_size, frame_size):
                frame = audio_data[i:i + frame_size]
                energy = np.sum(frame ** 2)
                frames.append(energy)
            
            # Look for significant energy changes that might indicate speaker changes
            if len(frames) < 3:
                return 1
            
            frames = np.array(frames)
            mean_energy = np.mean(frames)
            std_energy = np.std(frames)
            
            # Count significant energy transitions
            transitions = 0
            threshold = mean_energy + std_energy * 0.5
            
            for i in range(1, len(frames)):
                if abs(frames[i] - frames[i-1]) > threshold:
                    transitions += 1
            
            # Estimate speakers based on transitions and duration
            if duration < 30:  # Short recordings likely single speaker
                return 1
            elif transitions > duration * 0.3:  # Many transitions suggest multiple speakers
                return min(3, 2)  # Cap at 3 speakers for fallback
            else:
                return 1
                
        except Exception:
            return 1  # Default to single speaker on any error
    
    def _calculate_speaker_stats(self, segments: List[Dict], speakers: set) -> Dict:
        """Calculate speaking time statistics for each speaker"""
        stats = {}
        
        for speaker in speakers:
            speaker_segments = [s for s in segments if s["speaker"] == speaker]
            total_time = sum(s["duration"] for s in speaker_segments)
            num_segments = len(speaker_segments)
            avg_segment_length = total_time / num_segments if num_segments > 0 else 0
            
            stats[speaker] = {
                "total_speaking_time": round(total_time, 2),
                "num_segments": num_segments,
                "avg_segment_length": round(avg_segment_length, 2),
                "percentage_of_total": 0  # Will be calculated after all speakers
            }
        
        # Calculate percentages
        total_speaking_time = sum(stats[s]["total_speaking_time"] for s in stats)
        if total_speaking_time > 0:
            for speaker in stats:
                stats[speaker]["percentage_of_total"] = round(
                    (stats[speaker]["total_speaking_time"] / total_speaking_time) * 100, 1
                )
        
        return stats


class EnhancedAudioTranscriptionService(AudioTranscriptionService):
    """Enhanced audio service with speaker diarization support"""
    
    def __init__(self):
        super().__init__()
        self.diarization_service = SpeakerDiarizationService()
    
    def transcribe_with_speaker_analysis(self, audio_data: bytes) -> Dict:
        """Transcribe audio with speaker diarization and analysis"""
        
        # Get standard transcription and analysis
        base_result = self.transcribe_with_analysis(audio_data)
        
        if "error" in base_result:
            return base_result
        
        # Add speaker diarization
        try:
            speaker_info = self.diarization_service.detect_speakers(audio_data)
            
            # Merge speaker information with transcription
            enhanced_result = base_result.copy()
            enhanced_result["speaker_analysis"] = speaker_info
            
            # Add speaker-aware tips to analysis
            enhanced_result["analysis"]["speaker_feedback"] = self._generate_speaker_feedback(speaker_info)
            
            return enhanced_result
            
        except Exception as e:
            print(f"Speaker analysis failed, returning base transcription: {e}")
            # Add warning about speaker analysis failure
            base_result["speaker_analysis"] = {
                "error": str(e),
                "num_speakers": 1,
                "is_multi_speaker": False,
                "method": "failed"
            }
            return base_result
    
    def _generate_speaker_feedback(self, speaker_info: Dict) -> Dict:
        """Generate feedback based on speaker analysis"""
        feedback = {
            "multi_speaker_detected": speaker_info.get("is_multi_speaker", False),
            "confidence": speaker_info.get("confidence", "unknown"),
            "recommendations": []
        }
        
        num_speakers = speaker_info.get("num_speakers", 1)
        
        if num_speakers > 1:
            feedback["recommendations"].append({
                "type": "multi_speaker_warning",
                "title": "Multiple Speakers Detected",
                "description": f"Detected {num_speakers} speakers. For best evaluation results, consider individual recordings.",
                "priority": "medium"
            })
            
            # Check speaking balance
            speaker_stats = speaker_info.get("speaker_stats", {})
            if speaker_stats:
                percentages = [stats["percentage_of_total"] for stats in speaker_stats.values()]
                max_percentage = max(percentages) if percentages else 0
                
                if max_percentage > 80:
                    feedback["recommendations"].append({
                        "type": "unbalanced_speakers",
                        "title": "Unbalanced Speaking Time",
                        "description": "One speaker dominates the conversation. Encourage more balanced participation.",
                        "priority": "low"
                    })
        
        return feedback


# Global instances
audio_service = AudioTranscriptionService()
