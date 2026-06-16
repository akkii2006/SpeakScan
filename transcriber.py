import whisper


def load_whisper_model(model_size: str = "base"):
    return whisper.load_model(model_size)


def transcribe(audio_path: str, model) -> dict:
    result = model.transcribe(audio_path, task="transcribe")
    return result


def get_full_transcript(result: dict) -> str:
    return result["text"].strip()


def get_detected_language(result: dict) -> str:
    return result.get("language", "unknown")


def get_segments(result: dict) -> list[dict]:
    segments = []
    detected_language = result.get("language", "unknown")
    for seg in result["segments"]:
        segments.append({
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
            "language": seg.get("language", detected_language)
        })
    return segments
