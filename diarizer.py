from pyannote.audio import Pipeline
import torch


def diarize(audio_path: str, hf_token: str, vad_segments: list[dict] | None = None) -> list[dict]:
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


def export_rttm(diar_segments: list[dict], title: str, output_path: str):
    """Export diarization segments to RTTM format."""
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in diar_segments:
            duration = round(seg["end"] - seg["start"], 3)
            start = round(seg["start"], 3)
            speaker = seg["speaker"]
            # RTTM format: SPEAKER <file> <chnl> <tbeg> <tdur> <ortho> <stype> <name> <conf>
            f.write(f"SPEAKER {title} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA>\n")


def _overlap_ratio(start: float, end: float, speaker: str, diar_segments: list[dict]) -> float:
    duration = end - start
    if duration <= 0:
        return 0.0

    overlap_time = 0.0
    for d_seg in diar_segments:
        if d_seg["speaker"] == speaker:
            continue

        overlap_start = max(start, d_seg["start"])
        overlap_end = min(end, d_seg["end"])
        overlap_time += max(0.0, overlap_end - overlap_start)

    return overlap_time / duration


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
            "text": t_seg["text"],
            "language": t_seg.get("language", "unknown"),
            "overlap_ratio": round(_overlap_ratio(t_start, t_end, speaker, diar_segments), 3)
        })

    return merged