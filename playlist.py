import subprocess


PLAYLIST_INDICATORS = ["list=", "/playlist", "/channel/", "/c/", "/@"]


def is_playlist_url(url):
    return any(indicator in url for indicator in PLAYLIST_INDICATORS)


def expand_playlist(url):
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--print", "url",
        url
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to expand playlist: {result.stderr.strip()}")

    urls = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return urls