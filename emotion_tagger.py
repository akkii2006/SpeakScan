from transformers import pipeline


def load_emotion_classifier():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=1
    )


def tag_emotions(segments: list[dict], classifier) -> list[dict]:
    tagged = []

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            emotion = "neutral"
            score = 0.0
        else:
            result = classifier(text[:512])
            emotion = result[0][0]["label"].lower()
            score = round(result[0][0]["score"], 4)

        tagged.append({
            **seg,
            "emotion": emotion,
            "emotion_score": score
        })

    return tagged
