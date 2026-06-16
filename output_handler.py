import os
import json
import csv


def save_outputs(title: str, transcript: str, segments: list[dict], base_output_dir: str) -> str:
    video_dir = os.path.join(base_output_dir, title)
    os.makedirs(video_dir, exist_ok=True)

    with open(os.path.join(video_dir, "transcript.txt"), "w", encoding="utf-8") as f:
        f.write(transcript)

    with open(os.path.join(video_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

    if segments:
        with open(os.path.join(video_dir, "results.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=segments[0].keys())
            writer.writeheader()
            writer.writerows(segments)

    return video_dir


def update_results(video_dir: str, segments: list[dict]) -> str:
    results_path = os.path.join(video_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)
    return results_path


def is_already_processed(base_output_dir: str, title: str) -> bool:
    """Idempotency check — returns True if this video has already been fully processed."""
    video_dir = os.path.join(base_output_dir, title)
    required_files = ["results.json", "transcript.txt", "stats.json", "dataset.json"]
    return all(os.path.exists(os.path.join(video_dir, f)) for f in required_files)


def view_results(video_dir: str):
    transcript_path = os.path.join(video_dir, "transcript.txt")
    results_path = os.path.join(video_dir, "results.json")

    print("\n--- Transcript ---")
    if os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            print(f.read())
    else:
        print("No transcript found.")

    print("\n--- Segments (first 5) ---")
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for seg in data[:5]:
            print(f"[{seg['start']}s - {seg['end']}s] {seg.get('speaker', '?')} | {seg.get('emotion', '?')} | {seg.get('language', '?')} | {seg['text']}")
        if len(data) > 5:
            print(f"... and {len(data) - 5} more segments in results.json")
    else:
        print("No results found.")
