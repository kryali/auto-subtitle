import os
import warnings
from typing import Iterator, TextIO


def str2bool(string):
    string = string.lower()
    str2val = {"true": True, "false": False}

    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(
            f"Expected one of {set(str2val.keys())}, got {string}")


def format_timestamp(seconds: float, always_include_hours: bool = False):
    # Disallow missing timestamps to avoid silent errors
    if seconds is None:
        raise ValueError("Timestamp value cannot be None")
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def write_srt(transcript: Iterator, file: TextIO):
    """Write an iterator of Whisper segments to an opened SRT file handle.

    The *transcript* iterator may yield either dictionaries with the keys
    ``{"start", "end", "text"}`` or objects (e.g., ``faster_whisper.Segment``)
    exposing ``start``, ``end`` and ``text`` attributes. Handling both makes the
    rest of the codebase agnostic to the concrete Whisper implementation used.
    """

    for i, segment in enumerate(transcript, start=1):
        # Support both mapping and attribute-based segment representations
        if isinstance(segment, dict):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]
        else:
            start = segment.start
            end = segment.end
            text = segment.text

        # Skip segments with missing or illogical timestamps to avoid corrupt SRT entries
        if start is None or end is None:
            warnings.warn(
                "Skipping segment due to missing timestamp(s): "
                f"start={start}, end={end}"
            )
            continue

        if end < start:
            warnings.warn(
                "Skipping segment because end time precedes start time: "
                f"start={start}, end={end}"
            )
            continue

        print(
            f"{i}\n"
            f"{format_timestamp(start, always_include_hours=True)} --> "
            f"{format_timestamp(end, always_include_hours=True)}\n"
            f"{text.strip().replace('-->', '->')}\n",
            file=file,
            flush=True,
        )


def filename(path):
    return os.path.splitext(os.path.basename(path))[0]
