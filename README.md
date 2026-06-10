# SpeakScan

Audio pipeline that downloads YouTube videos, transcribes them, identifies speakers, tags emotions, and generates training data for speech and TTS models.

GitHub: https://github.com/akkii2006/SpeakScan

> Note: A full rewrite is currently in progress to improve usability, cross-platform support, and overall reliability. The current version works but may require manual setup steps depending on your environment.

---

## What it does

1. Downloads audio from a YouTube URL using yt-dlp
2. Transcribes the audio using Whisper
3. Runs speaker diarization using pyannote to identify who spoke when
4. Tags each segment with an emotion label using a wav2vec2 classifier trained on IEMOCAP
5. Saves a transcript, structured JSON, and CSV to an output folder per video
6. Generates waveform, mel spectrogram, and energy visualizations for the full audio
7. Exports an annotation dataset JSON
8. Generates a full TTS training dataset - per-segment mel spectrograms as .npy files paired with text, speaker, and emotion labels

---

## Use cases

**TTS model training** - The TTS dataset pairs each transcript segment with its mel spectrogram (.npy), speaker ID, and emotion label. This is the input-output format used by models like Tacotron and VITS: text in, spectrogram out. The pipeline generates the full dataset automatically, covering every segment in the audio with a corresponding spectrogram file ready for training.

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

The diarization pipeline uses gated models on HuggingFace. You need to visit each link and click "Accept" while logged into your HuggingFace account:

- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0
- https://huggingface.co/pyannote/speaker-diarization-community-1

Once accepted, generate a token at https://huggingface.co/settings/tokens. You will be prompted to enter it when you run the pipeline.

---

## Usage

```bash
python main.py
```

The pipeline will ask for your HuggingFace token, then prompt you for a YouTube URL. You can run it in single or batch mode. After processing, outputs are saved automatically.

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
    dataset.json              annotation dataset
    tts_dataset.json          TTS training dataset
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
    "end": 5.0,
    "speaker": "SPEAKER_00",
    "text": "So a couple days ago, I dropped a Twitter thread.",
    "emotion": "neu",
    "emotion_score": 1.0
  },
  {
    "start": 5.32,
    "end": 6.8,
    "speaker": "SPEAKER_00",
    "text": "Yes, I still call it Twitter.",
    "emotion": "neu",
    "emotion_score": 1.0
  }
]
```

### dataset.json format

```json
[
  {
    "audio_source": "video_title",
    "start": 0.0,
    "end": 5.0,
    "speaker": "SPEAKER_00",
    "text": "So a couple days ago, I dropped a Twitter thread.",
    "emotion": "neu",
    "emotion_score": 1.0
  },
  {
    "audio_source": "video_title",
    "start": 5.32,
    "end": 6.8,
    "speaker": "SPEAKER_00",
    "text": "Yes, I still call it Twitter.",
    "emotion": "neu",
    "emotion_score": 1.0
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
    "end": 5.0,
    "speaker": "SPEAKER_00",
    "text": "So a couple days ago, I dropped a Twitter thread.",
    "emotion": "neu",
    "emotion_score": 1.0,
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
- Emotion classification: [speechbrain/emotion-recognition-wav2vec2-IEMOCAP](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP)

---

## Notes
- MacOS users must run 'brew install ffmpeg yt-dlp' before running         
- GPU is used automatically if available
- Whisper model size can be changed in transcriber.py (tiny, base, small, medium, large)
- Mel spectrograms are generated at 22050Hz, 80 mel bands, matching standard TTS training configurations
- PyTorch must be installed matching your CUDA version - see https://pytorch.org/get-started/locally

---

## Output Images

![waveform](outputs/I%20Visited%20Apple's%20Secret%20iPhone%20Testing%20Labs!/waveform.png)
![Segment Energy](<outputs/I Visited Apple's Secret iPhone Testing Labs!/segment_energy.png>)
![Mel Spectrogram](<outputs/I Visited Apple's Secret iPhone Testing Labs!/mel_spectrogram.png>)
