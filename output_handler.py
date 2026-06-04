import os
import json
import csv


def save_outputs(title: str, transcript: str, segments: list[dict], base_output_dir: str) -> str:
    video_dir = os.path.join(base_output_dir, title)
    os.makedirs(video_dir, exist_ok=True)

    transcript_path = os.path.join(video_dir, "transcript.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    json_path = os.path.join(video_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(video_dir, "results.csv")
    if segments:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=segments[0].keys())
            writer.writeheader()
            writer.writerows(segments)

    return video_dir


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
            print(f"[{seg['start']}s - {seg['end']}s] {seg.get('speaker', '?')} | {seg.get('emotion', '?')} | {seg['text']}")
        if len(data) > 5:
            print(f"... and {len(data) - 5} more segments in results.json")
    else:
        print("No results found.")
