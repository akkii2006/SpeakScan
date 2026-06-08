import os
import threading
import itertools
import time
import json
import warnings
import logging

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["SPEECHBRAIN_VERBOSITY"] = "error"

from downloader import download_audio
from transcriber import transcribe, get_full_transcript, get_segments
from diarizer import diarize, merge_diarization_with_transcript
from emotion_tagger import load_emotion_classifier, tag_emotions
from output_handler import save_outputs, view_results
from dataset_converter import convert_to_dataset
from visualizer import generate_visualizations
from tts_dataset import generate_tts_dataset

OUTPUT_DIR = "outputs"
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def get_hf_token() -> str:
    token = HF_TOKEN
    if not token:
        token = input("Enter your HuggingFace token: ").strip()
    return token


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


def run_pipeline(url: str, hf_token: str, emotion_classifier):
    print()
    title, audio_path = run_with_spinner("Downloading audio", download_audio, url, OUTPUT_DIR)
    print(f"  Title: {title}")

    whisper_result = run_with_spinner("Transcribing", transcribe, audio_path)
    full_transcript = get_full_transcript(whisper_result)
    transcript_segments = get_segments(whisper_result)
    print(f"  Segments: {len(transcript_segments)}")

    diar_segments = run_with_spinner("Running speaker diarization", diarize, audio_path, hf_token)
    print(f"  Speakers found: {len(set(s['speaker'] for s in diar_segments))}")

    merged = merge_diarization_with_transcript(diar_segments, transcript_segments)
    tagged = run_with_spinner("Tagging emotions", tag_emotions, merged, emotion_classifier, audio_path)
    video_dir = os.path.join(OUTPUT_DIR, title)
    run_with_spinner("Saving outputs", save_outputs, title, full_transcript, tagged, OUTPUT_DIR)

    return title, audio_path, video_dir


def process_all(title: str, audio_path: str, video_dir: str):
    with open(os.path.join(video_dir, "results.json")) as f:
        segments = json.load(f)
    run_with_spinner("Generating visualizations", generate_visualizations, audio_path, video_dir, segments)
    convert_to_dataset(video_dir, title)
    dataset_path, count = run_with_spinner("Generating TTS dataset", generate_tts_dataset, audio_path, video_dir, title)
    print(f"  Done. {count} TTS segments saved to: {video_dir}")


def main():
    print("SpeakScan - Audio Pipeline")
    print("--------------------------")

    hf_token = get_hf_token()

    print("Loading emotion classifier...")
    emotion_classifier = load_emotion_classifier()
    print("Ready.\n")

    mode = input("Run mode - (s)ingle or (b)atch? ").strip().lower()

    if mode in ("batch", "b"):
        urls = []
        print("Enter YouTube URLs one by one. Type 'done' when finished:")
        i = 1
        while True:
            url = input(f"  URL {i}: ").strip()
            if url.lower() == "done":
                break
            if url:
                urls.append(url)
                i += 1

        if not urls:
            print("No URLs entered.")
            return

        print(f"\nProcessing {len(urls)} video(s)...")
        for idx, url in enumerate(urls, 1):
            print(f"\n[{idx}/{len(urls)}] {url}")
            try:
                title, audio_path, video_dir = run_pipeline(url, hf_token, emotion_classifier)
                process_all(title, audio_path, video_dir)
            except Exception as e:
                print(f"  Error: {e}. Skipping.")

    else:
        url = input("Enter YouTube URL: ").strip()
        if not url:
            print("No URL entered.")
            return
        try:
            title, audio_path, video_dir = run_pipeline(url, hf_token, emotion_classifier)
            process_all(title, audio_path, video_dir)
        except Exception as e:
            print(f"Error: {e}")

    print("\nExiting SpeakScan.")


if __name__ == "__main__":
    main()
