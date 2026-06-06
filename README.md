# SpeakScan

Audio pipeline that downloads YouTube videos, transcribes them, identifies speakers, tags emotions, and generates training data for speech and TTS models.

GitHub: https://github.com/akkii2006/SpeakScan

---

## What it does

1. Downloads audio from a YouTube URL using yt-dlp
2. Transcribes the audio using Whisper
3. Runs speaker diarization using pyannote to identify who spoke when
4. Tags each segment with an emotion label using a DistilRoBERTa classifier
5. Saves a transcript, structured JSON, and CSV to an output folder per video
6. Generates waveform, mel spectrogram, and energy visualizations for the full audio
7. Optionally exports an annotation dataset JSON
8. Optionally generates a TTS training dataset - per-segment mel spectrograms as .npy files paired with text, speaker, and emotion labels

---

## Use cases

**TTS model training** - Each segment in the TTS dataset contains the transcript text, speaker ID, emotion label, and a path to the mel spectrogram (.npy) for that exact audio slice. This is the input-output pair format used by models like Tacotron and VITS: text in, spectrogram out.

**ASR dataset construction** - The annotation dataset pairs timestamped transcripts with speaker labels and emotion tags, ready for fine-tuning ASR models on specific speaker styles or emotional speech.

**Speech research** - Waveform and spectrogram visualizations alongside per-segment energy make it easy to analyze speech patterns, pacing, and emotional variation across a recording.

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

The pipeline will ask for your HuggingFace token, then prompt you for a YouTube URL. After processing, you can:

- Generate waveform and spectrogram visualizations
- Export an annotation dataset JSON
- Generate a TTS training dataset with per-segment spectrograms
- View results in the terminal
- Process another video or exit

---

## Output structure

```
outputs/
  {video_title}/
    transcript.txt            full video transcript
    results.json              segments with speaker, emotion, timestamps
    results.csv               same data in CSV format
    waveform.png              full audio waveform with speaker segments color-coded
    mel_spectrogram.png       mel spectrogram of the full audio
    segment_energy.png        RMS energy per segment over time
    dataset.json              annotation dataset (only if selected)
    tts_dataset.json          TTS training dataset (only if selected)
    spectrograms/
      seg_0000.npy            mel spectrogram for segment 0
      seg_0001.npy            mel spectrogram for segment 1
      ...
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

### tts_dataset.json format

```json
[
  {
    "audio_source": "video_title",
    "segment_id": 0,
    "start": 0.0,
    "end": 4.5,
    "speaker": "SPEAKER_00",
    "text": "Hello and welcome.",
    "emotion": "neutral",
    "emotion_score": 0.91,
    "spectrogram": "outputs/video_title/spectrograms/seg_0000.npy",
    "sample_rate": 22050,
    "n_mels": 80
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
- Mel spectrograms are generated at 22050Hz, 80 mel bands, matching standard TTS training configurations
- PyTorch must be installed matching your CUDA version - see https://pytorch.org/get-started/locally
