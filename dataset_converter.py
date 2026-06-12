import os
import json

from text_normalizer import normalize_text


def convert_to_dataset(video_dir: str, title: str,
                        normalize_config: dict | None = None,
                        filter_quality: bool = True) -> str:
    results_path = os.path.join(video_dir, "results.json")

    if not os.path.exists(results_path):
        raise FileNotFoundError(f"results.json not found in {video_dir}")

    with open(results_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    normalize_config = normalize_config or {}

    dataset = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        if filter_quality and not seg.get("quality_ok", True):
            continue

        normalized_text = normalize_text(
            text,
            lowercase=normalize_config.get("lowercase", False),
            remove_punctuation=normalize_config.get("remove_punctuation", False),
            expand_nums=normalize_config.get("expand_numbers", False)
        )

        dataset.append({
            "audio_source": title,
            "start": seg["start"],
            "end": seg["end"],
            "speaker": seg.get("speaker", "UNKNOWN"),
            "text": normalized_text,
            "raw_text": text,
            "emotion": seg.get("emotion", "neutral"),
            "emotion_score": seg.get("emotion_score", 0.0),
            "quality_flags": seg.get("quality_flags", [])
        })

    dataset_path = os.path.join(video_dir, "dataset.json")
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    return dataset_path