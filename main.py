import os
import sys
import threading
import itertools
import time
import json
import warnings
import logging
import webbrowser
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["SPEECHBRAIN_VERBOSITY"] = "error"

from downloader import download_audio
from transcriber import transcribe, get_full_transcript, get_segments, load_whisper_model
from diarizer import diarize, merge_diarization_with_transcript
from emotion_tagger import load_emotion_classifier, tag_emotions
from output_handler import save_outputs, view_results, update_results
from dataset_converter import convert_to_dataset
from visualizer import generate_visualizations
from tts_dataset import generate_tts_dataset
from quality_filter import evaluate_segments
from stats import generate_stats_report
from keyword_extractor import generate_keywords_report
from report_generator import generate_report
from playlist import is_playlist_url, expand_playlist

OUTPUT_DIR = "outputs"
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def get_hf_token() -> str:
    token = HF_TOKEN
    if not token:
        if not sys.stdin.isatty():
            raise RuntimeError(
                "HF_TOKEN environment variable not set and no interactive "
                "terminal available to prompt for it."
            )
        token = input("Enter your HuggingFace token: ").strip()
    if not token:
        raise RuntimeError("HuggingFace token is required but was not provided.")
    return token


def is_first_run() -> bool:
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    whisper_cache = Path.home() / ".cache" / "whisper"

    emotion_model_cached = (
        hf_cache / "models--speechbrain--emotion-recognition-wav2vec2-IEMOCAP"
    ).exists()

    whisper_model_cached = whisper_cache.exists() and any(whisper_cache.iterdir())

    return not (emotion_model_cached and whisper_model_cached)


def spinner(label: str, stop_event: threading.Event):
    frames = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
    while not stop_event.is_set():
        print(f"\r{next(frames)} {label}...", end="", flush=True)
        time.sleep(0.1)
    print(f"\r✓ {label}    ")


def run_with_spinner(label: str, fn, *args):
    stop = threading.Event()
    t = threading.Thread(target=spinner, args=(label, stop))
    t.start()
    try:
        result = fn(*args)
    finally:
        stop.set()
        t.join()
    return result


def safe_run_with_spinner(label: str, fn, *args):
    try:
        return run_with_spinner(label, fn, *args)
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        raise RuntimeError(f"GPU out of memory during '{label}': {e}") from e
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            raise RuntimeError(f"GPU out of memory during '{label}': {e}") from e
        raise


def get_pipeline_config():
    print("\nPipeline configuration (press Enter for defaults):\n")

    quality_enabled = input("Enable audio quality filtering? (Y/n): ").strip().lower() != "n"

    print("\nText normalization for dataset exports:")
    lowercase = input("  Lowercase text? (y/N): ").strip().lower() == "y"
    remove_punct = input("  Remove punctuation? (y/N): ").strip().lower() == "y"
    expand_nums = input("  Expand numbers to words? (y/N): ").strip().lower() == "y"

    quality_config = {
        "enabled": quality_enabled,
        "min_snr_db": 5.0,
        "clipping_threshold": 0.99,
        "clipping_max_ratio": 0.001,
        "overlap_ratio": 0.3,
        "min_duration": 0.1
    }

    normalize_config = {
        "lowercase": lowercase,
        "remove_punctuation": remove_punct,
        "expand_numbers": expand_nums
    }

    return quality_config, normalize_config


def collect_urls(mode: str) -> list[str]:
    if mode in ("batch", "b"):
        raw_urls = []
        print("Enter YouTube URLs one by one. Type 'done' when finished:")
        i = 1
        while True:
            url = input(f"  URL {i}: ").strip()
            if url.lower() == "done":
                break
            if url:
                raw_urls.append(url)
                i += 1
        return raw_urls

    url = input("Enter YouTube URL: ").strip()
    return [url] if url else []


def expand_urls(raw_urls: list[str]) -> list[str]:
    expanded = []
    for url in raw_urls:
        if is_playlist_url(url):
            print(f"\nDetected playlist/channel URL: {url}")
            try:
                playlist_urls = run_with_spinner("Expanding playlist", expand_playlist, url)
                print(f"  Found {len(playlist_urls)} video(s)")
                expanded.extend(playlist_urls)
            except Exception as e:
                print(f"  Failed to expand playlist: {e}. Skipping.")
        else:
            expanded.append(url)
    return expanded


def run_pipeline(url: str, hf_token: str, emotion_classifier, whisper_model):
    print()
    title, audio_path = run_with_spinner("Downloading audio", download_audio, url, OUTPUT_DIR)
    print(f"  Title: {title}")

    whisper_result = safe_run_with_spinner("Transcribing", transcribe, audio_path, whisper_model)
    full_transcript = get_full_transcript(whisper_result)
    transcript_segments = get_segments(whisper_result)
    print(f"  Segments: {len(transcript_segments)}")

    diar_segments = safe_run_with_spinner("Running speaker diarization", diarize, audio_path, hf_token)
    print(f"  Speakers found: {len(set(s['speaker'] for s in diar_segments))}")

    merged = merge_diarization_with_transcript(diar_segments, transcript_segments)
    tagged = safe_run_with_spinner("Tagging emotions", tag_emotions, merged, emotion_classifier, audio_path)
    video_dir = os.path.join(OUTPUT_DIR, title)
    run_with_spinner("Saving outputs", save_outputs, title, full_transcript, tagged, OUTPUT_DIR)

    return title, audio_path, video_dir, diar_segments


def print_stats_summary(stats: dict):
    print(f"  Duration: {stats['total_duration_hours']:.2f} hrs over {stats['num_segments']} segments")
    print(f"  Speakers: {stats['num_speakers']}")

    for speaker, data in stats["speaker_stats"].items():
        print(f"    {speaker}: {data['talk_time_percent']}% talk time, {data['turn_count']} turns")

    top_emotions = sorted(stats["emotion_distribution"].items(), key=lambda x: x[1]["percent"], reverse=True)
    if top_emotions:
        summary = ", ".join(f"{e} ({d['percent']}%)" for e, d in top_emotions[:3])
        print(f"  Top emotions: {summary}")


def process_all(title: str, audio_path: str, video_dir: str, diar_segments: list[dict],
                 quality_config: dict, normalize_config: dict):
    with open(os.path.join(video_dir, "results.json"), encoding="utf-8") as f:
        segments = json.load(f)

    if quality_config["enabled"]:
        segments = safe_run_with_spinner(
            "Evaluating audio quality", evaluate_segments, audio_path, segments, diar_segments,
            quality_config["min_snr_db"], quality_config["clipping_threshold"],
            quality_config["clipping_max_ratio"], quality_config["overlap_ratio"],
            quality_config["min_duration"]
        )
        flagged = sum(1 for s in segments if not s["quality_ok"])
        print(f"  Quality check: {flagged}/{len(segments)} segment(s) flagged")
    else:
        for seg in segments:
            seg["quality_ok"] = True
            seg["quality_flags"] = []

    update_results(video_dir, segments)

    safe_run_with_spinner("Generating visualizations", generate_visualizations, audio_path, video_dir, segments)

    stats, _ = run_with_spinner("Computing dataset stats", generate_stats_report, video_dir)
    print_stats_summary(stats)

    filter_quality = quality_config["enabled"]
    run_with_spinner("Converting annotation dataset", convert_to_dataset, video_dir, title, normalize_config, filter_quality)

    dataset_path, count = safe_run_with_spinner(
        "Generating TTS dataset", generate_tts_dataset, audio_path, video_dir, title, normalize_config, filter_quality
    )
    print(f"  Done. {count} TTS segments saved to: {video_dir}")

    transcript_path = os.path.join(video_dir, "transcript.txt")
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read()

    keywords, _ = run_with_spinner("Extracting keywords", generate_keywords_report, transcript, video_dir)
    if keywords:
        top = ", ".join(k["keyword"] for k in keywords[:5])
        print(f"  Keywords: {top}")


def main():
    print("SpeakScan - Audio Pipeline")
    print("--------------------------")

    try:
        hf_token = get_hf_token()
    except RuntimeError as e:
        print(f"Fatal: {e}")
        return

    first_run = is_first_run()
    if first_run:
        print("\nFirst run detected - downloading model weights (~1GB total).")
        print("This is a one-time setup and may take several minutes depending")
        print("on your connection. Subsequent runs will start almost instantly.\n")
    else:
        print("\nLoading models...\n")

    try:
        print("Loading emotion classifier...")
        emotion_classifier = load_emotion_classifier()

        print("Loading Whisper model...")
        whisper_model = load_whisper_model()
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        print(f"Fatal: GPU out of memory while loading models: {e}")
        return
    except Exception as e:
        print(f"Fatal: failed to load models: {e}")
        return

    print("Ready.\n")

    mode = input("Run mode - (s)ingle or (b)atch? ").strip().lower()

    raw_urls = collect_urls(mode)
    if not raw_urls:
        print("No URLs entered.")
        return

    urls = expand_urls(raw_urls)
    if not urls:
        print("No videos to process.")
        return

    quality_config, normalize_config = get_pipeline_config()

    print(f"\nProcessing {len(urls)} video(s)...")
    processed = []
    for idx, url in enumerate(urls, 1):
        print(f"\n[{idx}/{len(urls)}] {url}")
        try:
            title, audio_path, video_dir, diar_segments = run_pipeline(url, hf_token, emotion_classifier, whisper_model)
            process_all(title, audio_path, video_dir, diar_segments, quality_config, normalize_config)
            processed.append((title, video_dir))
        except Exception as e:
            print(f"  Error: {e}. Skipping.")

    if processed:
        open_browser = input("\nOpen results in browser? (y/N): ").strip().lower() == "y"
        for title, video_dir in processed:
            try:
                report_path = generate_report(video_dir, title)
                if open_browser:
                    webbrowser.open(f"file://{os.path.abspath(report_path)}")
            except Exception as e:
                print(f"  Failed to generate report for {title}: {e}")

    print("\nExiting SpeakScan.")


if __name__ == "__main__":
    main()