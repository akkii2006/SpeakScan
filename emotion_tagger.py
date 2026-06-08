import torch
import torchaudio
import numpy as np
from speechbrain.inference.interfaces import foreign_class


def load_emotion_classifier():
    classifier = foreign_class(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier"
    )
    return classifier


def tag_emotions(segments: list[dict], classifier, audio_path: str, sr: int = 16000) -> list[dict]:
    waveform, orig_sr = torchaudio.load(audio_path)

    if orig_sr != sr:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=sr)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    tagged = []

    for seg in segments:
        text = seg.get("text", "").strip()

        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        chunk = waveform[:, start_sample:end_sample]

        if chunk.shape[1] < sr * 0.1:
            tagged.append({**seg, "emotion": "neutral", "emotion_score": 0.0})
            continue

        try:
            out_prob, score, index, label = classifier.classify_batch(chunk)
            emotion = label[0].lower()
            emotion_score = round(score[0].item(), 4)
        except Exception:
            emotion = "neutral"
            emotion_score = 0.0

        tagged.append({
            **seg,
            "emotion": emotion,
            "emotion_score": emotion_score
        })

    return tagged
