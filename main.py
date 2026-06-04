import os
import sys
import threading
import itertools
import time
from downloader import download_audio
from transcriber import transcribe, get_full_transcript, get_segments
from diarizer import diarize, merge_diarization_with_transcript
from emotion_tagger import load_emotion_classifier, tag_emotions
from output_handler import save_outputs, view_results
from dataset_converter import convert_to_dataset

OUTPUT_DIR = "outputs"
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def get_hf_token() -> str:
    token = HF_TOKEN
    if not token:
        token = input("Enter your HuggingFace token (needed for diarization): ").strip()
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

    tagged = run_with_spinner("Tagging emotions", tag_emotions, merged, emotion_classifier)

    video_dir = run_with_spinner("Saving outputs", save_outputs, title, full_transcript, tagged, OUTPUT_DIR)

    print(f"\nDone. Results saved to: {video_dir}")
    return title, video_dir


def main():
    print("SpeakScan - Audio Pipeline")
    print("--------------------------")

    hf_token = get_hf_token()

    print("Loading emotion classifier...")
    emotion_classifier = load_emotion_classifier()
    print("Ready.\n")

    while True:
        url = input("Enter YouTube URL: ").strip()
        if not url:
            print("No URL entered. Try again.")
            continue

        try:
            title, video_dir = run_pipeline(url, hf_token, emotion_classifier)
        except Exception as e:
            print(f"Error: {e}")
            retry = input("Try another video? (y/n): ").strip().lower()
            if retry != "y":
                break
            continue

        convert = input("\nConvert results to training dataset JSON? (y/n): ").strip().lower()
        if convert == "y":
            try:
                dataset_path = convert_to_dataset(video_dir, title)
                print(f"Dataset saved to: {dataset_path}")
            except Exception as e:
                print(f"Dataset conversion failed: {e}")

        view = input("\nView results now? (y/n): ").strip().lower()
        if view == "y":
            view_results(video_dir)

        again = input("\nProcess another video? (y/n): ").strip().lower()
        if again != "y":
            print("Exiting SpeakScan.")
            break


if __name__ == "__main__":
    main()
