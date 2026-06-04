from pyannote.audio import Pipeline
import torch


def diarize(audio_path: str, hf_token: str) -> list[dict]:
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token
    )

    if torch.cuda.is_available():
        pipeline = pipeline.to(torch.device("cuda"))

    output = pipeline(audio_path)
    annotation = output.speaker_diarization

    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
            "speaker": speaker
        })

    return segments


def merge_diarization_with_transcript(diar_segments: list[dict], transcript_segments: list[dict]) -> list[dict]:
    merged = []

    for t_seg in transcript_segments:
        t_start = t_seg["start"]
        t_end = t_seg["end"]
        speaker = "UNKNOWN"
        best_overlap = 0.0

        for d_seg in diar_segments:
            overlap_start = max(t_start, d_seg["start"])
            overlap_end = min(t_end, d_seg["end"])
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                speaker = d_seg["speaker"]

        merged.append({
            "start": t_start,
            "end": t_end,
            "speaker": speaker,
            "text": t_seg["text"]
        })

    return merged
