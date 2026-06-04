import os
import json


def convert_to_dataset(video_dir: str, title: str) -> str:
    results_path = os.path.join(video_dir, "results.json")

    if not os.path.exists(results_path):
        raise FileNotFoundError(f"results.json not found in {video_dir}")

    with open(results_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    dataset = []
    for seg in segments:
        if not seg.get("text", "").strip():
            continue
        dataset.append({
            "audio_source": title,
            "start": seg["start"],
            "end": seg["end"],
            "speaker": seg.get("speaker", "UNKNOWN"),
            "text": seg["text"],
            "emotion": seg.get("emotion", "neutral"),
            "emotion_score": seg.get("emotion_score", 0.0)
        })

    dataset_path = os.path.join(video_dir, "dataset.json")
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    return dataset_path
