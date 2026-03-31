import whisper
from config import WHISPER_MODEL
from src.processing.transcript_segments import build_segment, segments_to_text

model = whisper.load_model(WHISPER_MODEL)

def transcribe_audio(audio_path):
    result = model.transcribe(audio_path)
    segments = []

    for raw_segment in result.get("segments", []):
        segment = build_segment(
            raw_segment.get("start"),
            raw_segment.get("end"),
            raw_segment.get("text", ""),
        )
        if segment:
            segments.append(segment)

    transcript_text = segments_to_text(segments) if segments else result.get("text", "").strip()
    return transcript_text, segments
