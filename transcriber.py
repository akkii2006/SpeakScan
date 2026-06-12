import whisper


def load_whisper_model(model_size: str = "base"):
    return whisper.load_model(model_size)


def transcribe(audio_path: str, model) -> dict:
    result = model.transcribe(audio_path)
    return result


def get_full_transcript(result: dict) -> str:
    return result["text"].strip()


def get_segments(result: dict) -> list[dict]:
    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip()
        })
    return segments