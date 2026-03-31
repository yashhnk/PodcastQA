import pytest
from unittest.mock import patch, MagicMock
from src.ingestion.youtube import fetch_youtube_transcript, get_video_info
from src.processing.transcript_segments import parse_vtt_segments, segments_to_timestamped_text

@patch('src.ingestion.youtube.yt_dlp.YoutubeDL')
def test_get_video_info(mock_ytdl):
    mock_instance = MagicMock()
    mock_ytdl.return_value.__enter__.return_value = mock_instance
    mock_instance.extract_info.return_value = {
        'title': 'Test Video',
        'duration': 120,
        'channel': 'Test Channel',
        'subtitles': {'en': {}},
        'automatic_captions': {}
    }

    info = get_video_info('http://test.url')
    assert info is not None
    assert info['title'] == 'Test Video'
    assert info['has_manual_subs'] == True
    assert info['has_auto_subs'] == False

@patch('src.ingestion.youtube.tempfile.TemporaryDirectory')
@patch('src.ingestion.youtube.yt_dlp.YoutubeDL')
def test_fetch_youtube_transcript_no_subs(mock_ytdl, mock_temp):
    mock_instance = MagicMock()
    mock_ytdl.return_value.__enter__.return_value = mock_instance
    mock_instance.extract_info.return_value = {
        'subtitles': {},
        'automatic_captions': {}
    }

    transcript, source, segments = fetch_youtube_transcript('http://test.url')
    assert transcript is None
    assert source is None
    assert segments == []


def test_parse_vtt_segments_creates_timestamped_segments():
    vtt_content = """WEBVTT

00:00:01.000 --> 00:00:03.000
Hello there

00:00:03.000 --> 00:00:06.000
General Kenobi
"""

    segments = parse_vtt_segments(vtt_content)

    assert segments == [
        {"start": 1.0, "end": 3.0, "timestamp": "00:01", "text": "Hello there"},
        {"start": 3.0, "end": 6.0, "timestamp": "00:03", "text": "General Kenobi"},
    ]
    assert segments_to_timestamped_text(segments) == "[00:01] Hello there\n[00:03] General Kenobi"
