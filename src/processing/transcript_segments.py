import re
from typing import Dict, List, Optional


TIMESTAMP_LINE_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})\s+-->\s+"
    r"(?P<end>\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})"
)


def parse_vtt_timestamp(value: str) -> float:
    parts = value.split(":")
    if len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    else:
        hours, minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def format_timestamp(seconds: Optional[float]) -> str:
    if seconds is None:
        return "00:00"

    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def normalize_segment_text(text: str) -> str:
    cleaned = re.sub(r"<[^>]+>", "", text)
    return " ".join(cleaned.split()).strip()


def build_segment(start: Optional[float], end: Optional[float], text: str) -> Optional[Dict[str, object]]:
    normalized_text = normalize_segment_text(text)
    if not normalized_text:
        return None

    return {
        "start": start,
        "end": end,
        "timestamp": format_timestamp(start),
        "text": normalized_text,
    }


def parse_vtt_segments(vtt_content: str) -> List[Dict[str, object]]:
    segments: List[Dict[str, object]] = []
    current_start: Optional[float] = None
    current_end: Optional[float] = None
    current_lines: List[str] = []

    def flush_current_segment() -> None:
        nonlocal current_start, current_end, current_lines
        if current_lines:
            segment = build_segment(current_start, current_end, " ".join(current_lines))
            if segment and (not segments or segment["text"] != segments[-1]["text"]):
                segments.append(segment)
        current_start = None
        current_end = None
        current_lines = []

    for raw_line in vtt_content.splitlines():
        line = raw_line.strip()

        if not line:
            flush_current_segment()
            continue

        if line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
            continue

        match = TIMESTAMP_LINE_RE.match(line)
        if match:
            flush_current_segment()
            current_start = parse_vtt_timestamp(match.group("start"))
            current_end = parse_vtt_timestamp(match.group("end"))
            continue

        if line.isdigit():
            continue

        current_lines.append(line)

    flush_current_segment()
    return segments


def segments_to_text(segments: List[Dict[str, object]]) -> str:
    return " ".join(segment["text"] for segment in segments if segment.get("text")).strip()


def segments_to_timestamped_text(segments: List[Dict[str, object]]) -> str:
    return "\n".join(
        f"[{segment['timestamp']}] {segment['text']}"
        for segment in segments
        if segment.get("text")
    )
