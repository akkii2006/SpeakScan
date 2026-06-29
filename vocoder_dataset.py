import os
import json
import numpy as np
import librosa
import soundfile as sf


def generate_vocoder_dataset(audio_path: str, video_dir: str, title: str,
                              filter_quality: bool = True) -> tuple[str, int]:
    results_path = os.path.join(video_dir, "results.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"results.json not found in {video_dir}")

    spec_dir = os.path.join(video_dir, "spectrograms")
    if not os.path.exists(spec_dir):
        raise FileNotFoundError(f"spectrograms directory not found in {video_dir}, run TTS dataset generation first")

    with open(results_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    y, sr = librosa.load(audio_path, sr=22050)

    wav_dir = os.path.join(video_dir, "wav_chunks")
    os.makedirs(wav_dir, exist_ok=True)

    dataset = []
    for i, seg in enumerate(segments):
        text = seg.get("text", "").strip()
        if not text:
            continue

        if filter_quality and not seg.get("quality_ok", True):
            continue

        spec_filename = f"seg_{i:04d}.npy"
        spec_path = os.path.join(spec_dir, spec_filename)
        if not os.path.exists(spec_path):
            continue

        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        chunk = y[start_sample:end_sample]

        if len(chunk) < sr * 0.1:
            continue

        wav_filename = f"seg_{i:04d}.wav"
        wav_path = os.path.join(wav_dir, wav_filename)
        sf.write(wav_path, chunk, sr)

        dataset.append({
            "audio_source": title,
            "segment_id": i,
            "start": seg["start"],
            "end": seg["end"],
            "speaker": seg.get("speaker", "UNKNOWN"),
            "language": seg.get("language", "unknown"),
            "emotion": seg.get("emotion", "neutral"),
            "emotion_score": seg.get("emotion_score", 0.0),
            "quality_flags": seg.get("quality_flags", []),
            "spectrogram": spec_path,
            "wav_chunk": wav_path,
            "sample_rate": sr,
            "n_mels": 80
        })

    dataset_path = os.path.join(video_dir, "vocoder_dataset.json")
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    return dataset_path, len(dataset)
