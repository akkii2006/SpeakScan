# SpeakScan

Audio pipeline that downloads YouTube videos, transcribes them, identifies speakers, tags emotions, detects language, and generates training data for speech and TTS models.

GitHub: https://github.com/akkii2006/SpeakScan

---

## What it does

1. Downloads audio from a YouTube URL (or expands a playlist/channel URL into individual videos) using yt-dlp
2. Runs Voice Activity Detection (VAD) using silero-vad to identify speech regions and filter out silence before transcription and diarization
3. Transcribes the audio using Whisper, with per-segment language detection
4. Runs speaker diarization using pyannote to identify who spoke when, and exports an RTTM file
5. Tags each segment with an emotion label using a wav2vec2 classifier trained on IEMOCAP
6. Evaluates audio quality per segment (SNR, clipping, overlapping speech) and flags low-quality segments
7. Saves a transcript, structured JSON, and CSV to an output folder per video
8. Generates waveform, mel spectrogram, segment energy, and emotion timeline visualizations for the full audio
9. Computes dataset stats: total duration, per-speaker talk time and turn counts, emotion distribution, language distribution
10. Extracts keywords/topics from the transcript
11. Exports an annotation dataset JSON, with optional text normalization (lowercase, punctuation removal, number expansion)
12. Generates a full TTS training dataset - per-segment mel spectrograms as .npy files paired with text, speaker, emotion, and language labels, with optional quality filtering
13. Optionally generates a vocoder training dataset - pairs each mel spectrogram with its corresponding audio chunk as a .wav file
14. Builds a self-contained HTML report combining the transcript, visualizations, stats, and keywords, which can be opened in your browser automatically
15. Skips already-processed videos automatically (idempotency) and retries failed steps with exponential backoff

---

## Use cases

**TTS model training** - The TTS dataset pairs each transcript segment with its mel spectrogram (.npy), speaker ID, emotion label, and language. This is the input-output format used by models like Tacotron and VITS: text in, spectrogram out. The pipeline generates the full dataset automatically, covering every segment in the audio with a corresponding spectrogram file ready for training. Low-quality segments (low SNR, clipping, heavy speaker overlap) can be filtered out automatically so they don't degrade training.

**Vocoder training** - The vocoder dataset pairs each mel spectrogram with its corresponding .wav audio chunk. This is the input-output format used by vocoders like HiFi-GAN and WaveGlow: spectrogram in, waveform out. Generated on demand so compute is not wasted if not needed.

**ASR dataset construction** - The annotation dataset pairs timestamped transcripts with speaker labels, emotion tags, and language codes, ready for fine-tuning ASR models on specific speaker styles, emotional speech, or multilingual content. Optional text normalization (lowercase, punctuation stripping, number expansion) lets you match the text format your training pipeline expects.

**Multilingual speech data** - Per-segment language detection via Whisper makes it easy to identify and separate segments by language across mixed-language recordings — useful for Indian content where speakers switch between Hindi, English, Tamil, and other languages.

**Speech research** - Waveform, spectrogram, segment energy, and emotion timeline visualizations make it easy to analyze speech patterns, pacing, and emotional variation across a recording. Speaker talk-time stats, emotion distributions, and language breakdowns give a quick read on conversation dynamics.

**Playlist/channel processing** - Pass a playlist or channel URL and SpeakScan will expand it into individual videos and process them all in batch. Already-processed videos are skipped automatically on re-runs, and failed videos are retried up to 3 times before being skipped.

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

Once accepted, generate a token at https://huggingface.co/settings/tokens. You can set it as an environment variable (`HF_TOKEN`), enter it when prompted and choose to save it, or use `--set-token` to update it at any time.

```bash
python main.py --set-token
```

The token is saved to a `.env` file in the project root. Make sure `.env` is in your `.gitignore`.

---

## Usage

```bash
python main.py
```

The pipeline will:

1. Load your HuggingFace token from the environment, `.env` file, or prompt you to enter it (with an option to save it for future runs)
2. Load the emotion classifier, Whisper model, and VAD model (first run downloads ~1GB of weights, one-time cost)
3. Ask for single or batch mode
4. Prompt for a YouTube URL (or URLs in batch mode) - playlist/channel URLs are automatically expanded into individual videos
5. Ask for pipeline configuration:
   - Enable audio quality filtering (default: yes)
   - Enable VAD pre-processing (default: yes)
   - Lowercase text, remove punctuation, expand numbers for dataset exports (default: no)
6. Process each video through the full pipeline — already-processed videos are skipped automatically
7. Ask whether to generate a vocoder dataset for each processed video
8. Ask whether to open the generated HTML report(s) in your browser

---

## Output structure

```
outputs/
  {video_title}/
    transcript.txt            full video transcript
    results.json              segments with speaker, emotion, language, timestamps, quality flags
    results.csv               same data in CSV format
    {title}.rttm              diarization output in standard RTTM format
    waveform.png              full audio waveform with speaker segments color-coded
    mel_spectrogram.png       mel spectrogram of the full audio
    segment_energy.png        RMS energy per segment over time
    emotion_timeline.png      emotion per segment over time, sized by confidence, colored by speaker
    stats.json                dataset stats, speaker talk-time, emotion distribution, language distribution, emotion timeline data
    keywords.json             extracted keywords/topics
    dataset.json              annotation dataset
    tts_dataset.json          TTS training dataset
    vocoder_dataset.json      vocoder training dataset (if generated)
    report.html               self-contained HTML report (transcript + visualizations + stats + keywords)
    spectrograms/
      seg_0000.npy            mel spectrogram for segment 0
      seg_0001.npy            mel spectrogram for segment 1
      ...
    wav_chunks/               (if vocoder dataset was generated)
      seg_0000.wav            audio chunk for segment 0
      seg_0001.wav            audio chunk for segment 1
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
    "language": "en",
    "emotion": "neu",
    "emotion_score": 1.0,
    "overlap_ratio": 0.0,
    "snr_db": 18.4,
    "quality_flags": [],
    "quality_ok": true
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
    "text": "so a couple days ago i dropped a twitter thread",
    "raw_text": "So a couple days ago, I dropped a Twitter thread.",
    "emotion": "neu",
    "emotion_score": 1.0,
    "language": "en",
    "quality_flags": []
  }
]
```

`text` reflects whatever normalization options were chosen (lowercase, punctuation removal, number expansion). `raw_text` always preserves the original transcribed text.

### tts_dataset.json format

```json
[
  {
    "audio_source": "video_title",
    "segment_id": 0,
    "start": 0.0,
    "end": 5.0,
    "speaker": "SPEAKER_00",
    "text": "so a couple days ago i dropped a twitter thread",
    "raw_text": "So a couple days ago, I dropped a Twitter thread.",
    "emotion": "neu",
    "emotion_score": 1.0,
    "language": "en",
    "quality_flags": [],
    "spectrogram": "outputs/video_title/spectrograms/seg_0000.npy",
    "sample_rate": 22050,
    "n_mels": 80
  }
]
```

### vocoder_dataset.json format

```json
[
  {
    "audio_source": "video_title",
    "segment_id": 0,
    "start": 0.0,
    "end": 5.0,
    "speaker": "SPEAKER_00",
    "language": "en",
    "emotion": "neu",
    "emotion_score": 1.0,
    "quality_flags": [],
    "spectrogram": "outputs/video_title/spectrograms/seg_0000.npy",
    "wav_chunk": "outputs/video_title/wav_chunks/seg_0000.wav",
    "sample_rate": 22050,
    "n_mels": 80
  }
]
```

If audio quality filtering is enabled, segments flagged with quality issues (low SNR, clipping, or excessive speaker overlap) are excluded from `dataset.json`, `tts_dataset.json`, and `vocoder_dataset.json`, but remain visible (with their flags) in `results.json`.

### stats.json format

```json
{
  "total_duration_seconds": 1840.5,
  "total_duration_hours": 0.51,
  "num_segments": 627,
  "avg_segment_length_seconds": 2.93,
  "num_speakers": 3,
  "emotion_distribution": {
    "neu": { "count": 301, "percent": 48.01 },
    "ang": { "count": 254, "percent": 40.51 },
    "hap": { "count": 72, "percent": 11.48 }
  },
  "language_distribution": {
    "en": { "count": 580, "percent": 92.5 },
    "hi": { "count": 47, "percent": 7.5 }
  },
  "speaker_stats": {
    "SPEAKER_00": { "talk_time_seconds": 1414.5, "talk_time_percent": 76.85, "turn_count": 47 },
    "SPEAKER_01": { "talk_time_seconds": 425.3, "talk_time_percent": 23.1, "turn_count": 46 },
    "UNKNOWN": { "talk_time_seconds": 0.9, "talk_time_percent": 0.05, "turn_count": 1 }
  },
  "emotion_by_speaker": { ... },
  "emotion_timeline": [ ... ]
}
```

### keywords.json format

```json
[
  { "keyword": "Apple", "score": 0.0123 },
  { "keyword": "iphone", "score": 0.0156 },
  { "keyword": "job", "score": 0.0341 }
]
```

Lower YAKE scores indicate more relevant keywords.

---

## Models used

- Transcription + language detection: [openai/whisper](https://github.com/openai/whisper) (base by default)
- Voice activity detection: [silero-vad](https://github.com/snakers4/silero-vad)
- Diarization: [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- Emotion classification: [speechbrain/emotion-recognition-wav2vec2-IEMOCAP](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP)
- Keyword extraction: [YAKE](https://github.com/LIAAD/yake)

---

## Notes

- MacOS users must run `brew install ffmpeg yt-dlp` before running
- GPU is used automatically if available
- Whisper model size can be changed in transcriber.py (tiny, base, small, medium, large)
- Mel spectrograms are generated at 22050Hz, 80 mel bands, matching standard TTS training configurations
- PyTorch must be installed matching your CUDA version - see https://pytorch.org/get-started/locally
- "UNKNOWN" speaker labels can appear for segments where no diarized speech overlaps the transcript segment
- To reprocess a video that was previously completed, delete its folder under `outputs/` and re-run

---

## Output Images

![Waveform](<outputs/I Found North Korean Spies on Discord…/waveform.png>)
![Segment Energy](<outputs/I Found North Korean Spies on Discord…/segment_energy.png>)
![Mel Spectrogram](<outputs/I Found North Korean Spies on Discord…/mel_spectrogram.png>)
![Emotion Timeline](<outputs/I Found North Korean Spies on Discord…/emotion_timeline.png>)
