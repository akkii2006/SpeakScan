import os
import json
import numpy as np
import librosa

from text_normalizer import normalize_text


def generate_tts_dataset(audio_path: str, video_dir: str, title: str,
                          normalize_config: dict | None = None,
                          filter_quality: bool = True) -> tuple[str, int]:
    results_path = os.path.join(video_dir, "results.json")

    if not os.path.exists(results_path):
        raise FileNotFoundError(f"results.json not found in {video_dir}")

    with open(results_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    normalize_config = normalize_config or {}

    y, sr = librosa.load(audio_path, sr=22050)

    spec_dir = os.path.join(video_dir, "spectrograms")
    os.makedirs(spec_dir, exist_ok=True)

    dataset = []
    for i, seg in enumerate(segments):
        text = seg.get("text", "").strip()
        if not text:
            continue

        if filter_quality and not seg.get("quality_ok", True):
            continue

        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        chunk = y[start_sample:end_sample]

        if len(chunk) < sr * 0.1:
            continue

        mel = librosa.feature.melspectrogram(
            y=chunk,
            sr=sr,
            n_mels=80,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            fmax=8000
        )
        mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

        spec_filename = f"seg_{i:04d}.npy"
        spec_path = os.path.join(spec_dir, spec_filename)
        np.save(spec_path, mel_db)

        normalized_text = normalize_text(
            text,
            lowercase=normalize_config.get("lowercase", False),
            remove_punctuation=normalize_config.get("remove_punctuation", False),
            expand_nums=normalize_config.get("expand_numbers", False)
        )

        dataset.append({
            "audio_source": title,
            "segment_id": i,
            "start": seg["start"],
            "end": seg["end"],
            "speaker": seg.get("speaker", "UNKNOWN"),
            "text": normalized_text,
            "raw_text": text,
            "emotion": seg.get("emotion", "neutral"),
            "emotion_score": seg.get("emotion_score", 0.0),
            "language": seg.get("language", "unknown"),
            "quality_flags": seg.get("quality_flags", []),
            "spectrogram": spec_path,
            "sample_rate": sr,
            "n_mels": 80
        })

    dataset_path = os.path.join(video_dir, "tts_dataset.json")
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    return dataset_path, len(dataset)
