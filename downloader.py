import os
import subprocess


def download_audio(url: str, output_dir: str) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    info_cmd = [
        "yt-dlp",
        "--print", "title",
        "--no-playlist",
        url
    ]
    result = subprocess.run(info_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to fetch video info: {result.stderr.strip()}")

    title = result.stdout.strip().replace("/", "_").replace("\\", "_")
    audio_path = os.path.join(output_dir, f"{title}.wav")

    dl_cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", audio_path,
        "--no-playlist",
        url
    ]
    dl_result = subprocess.run(dl_cmd, capture_output=True, text=True)
    if dl_result.returncode != 0:
        raise RuntimeError(f"Download failed: {dl_result.stderr.strip()}")

    return title, audio_path
