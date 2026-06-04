# SpeakScan

Audio pipeline that downloads YouTube videos, transcribes them, identifies speakers, tags emotions, and outputs structured data for model training.

GitHub: https://github.com/akkii2006/SpeakScan

---

## What it does

1. Downloads audio from a YouTube URL using yt-dlp
2. Transcribes the audio using Whisper
3. Runs speaker diarization using pyannote to identify who spoke when
4. Tags each segment with an emotion label using a DistilRoBERTa classifier
5. Saves a transcript, structured JSON, and CSV to an output folder per video
6. Optionally converts results into a training dataset JSON

---

## Setup

Install all dependencies via pip:

```bash
pip install -r requirements.txt
```

---

## HuggingFace Access

The diarization pipeline uses three gated models on HuggingFace. You need to visit each link and click "Accept" while logged into your HuggingFace account:

- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0
- https://huggingface.co/pyannote/speaker-diarization-community-1

Once accepted, generate a token at https://huggingface.co/settings/tokens. You will be prompted to enter it when you run the pipeline.

---

## Usage

```bash
python main.py
```

The pipeline will ask for your HuggingFace token, then prompt you for a YouTube URL. After processing, you can choose to convert results to a training dataset JSON, view the output, and process another video or exit.

---

## Output structure

```
outputs/
  {video_title}/
    transcript.txt       full video transcript
    results.json         segments with speaker, emotion, timestamps
    results.csv          same data in CSV format
    dataset.json         training dataset format (only if selected)
```

### results.json format

```json
[
  {
    "start": 0.0,
    "end": 4.5,
    "speaker": "SPEAKER_00",
    "text": "Hello and welcome.",
    "emotion": "neutral",
    "emotion_score": 0.91
  }
]
```

### dataset.json format

```json
[
  {
    "audio_source": "video_title",
    "start": 0.0,
    "end": 4.5,
    "speaker": "SPEAKER_00",
    "text": "Hello and welcome.",
    "emotion": "neutral",
    "emotion_score": 0.91
  }
]
```

---

## Models used

- Transcription: [openai/whisper](https://github.com/openai/whisper) (base by default)
- Diarization: [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- Emotion classification: [j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)

---

## Notes

- GPU is used automatically if available
- Whisper model size can be changed in transcriber.py (tiny, base, small, medium, large)
- PyTorch must be installed matching your CUDA version — see https://pytorch.org/get-started/locally
