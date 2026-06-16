import torch
import torchaudio


def load_vad_model():
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True
    )
    return model, utils


def get_speech_segments(audio_path: str, model, utils, min_silence_duration_ms: int = 500) -> list[dict]:
    get_speech_timestamps, _, read_audio, *_ = utils

    wav = read_audio(audio_path, sampling_rate=16000)

    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=16000,
        min_silence_duration_ms=min_silence_duration_ms,
        return_seconds=True
    )

    return [
        {"start": round(seg["start"], 2), "end": round(seg["end"], 2)}
        for seg in speech_timestamps
    ]


def filter_segments_by_vad(segments: list[dict], vad_segments: list[dict], min_overlap: float = 0.5) -> list[dict]:
    """Filter transcript/diarization segments to only those with sufficient VAD overlap."""
    filtered = []
    for seg in segments:
        seg_duration = seg["end"] - seg["start"]
        if seg_duration <= 0:
            continue

        overlap = 0.0
        for v in vad_segments:
            overlap_start = max(seg["start"], v["start"])
            overlap_end = min(seg["end"], v["end"])
            overlap += max(0.0, overlap_end - overlap_start)

        if (overlap / seg_duration) >= min_overlap:
            filtered.append(seg)

    return filtered
