import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def generate_visualizations(audio_path: str, output_dir: str, segments: list[dict]):
    os.makedirs(output_dir, exist_ok=True)

    y, sr = librosa.load(audio_path, sr=None)

    _plot_waveform(y, sr, segments, output_dir)
    _plot_mel_spectrogram(y, sr, output_dir)
    _plot_segment_energy(y, sr, segments, output_dir)


def _plot_waveform(y, sr, segments, output_dir):
    fig, ax = plt.subplots(figsize=(14, 4))
    times = np.linspace(0, len(y) / sr, len(y))
    ax.plot(times, y, linewidth=0.5, color='steelblue', alpha=0.8)

    colors = plt.cm.tab10.colors
    speakers = list({s["speaker"] for s in segments})
    for seg in segments:
        color = colors[speakers.index(seg["speaker"]) % len(colors)]
        ax.axvspan(seg["start"], seg["end"], alpha=0.15, color=color, label=seg["speaker"])

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i % len(colors)], alpha=0.4) for i, s in enumerate(speakers)]
    ax.legend(handles, speakers, loc="upper right", fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform with Speaker Segments")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "waveform.png"), dpi=150)
    plt.close()


def _plot_mel_spectrogram(y, sr, output_dir):
    fig, ax = plt.subplots(figsize=(14, 4))
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel Spectrogram")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mel_spectrogram.png"), dpi=150)
    plt.close()


def _plot_segment_energy(y, sr, segments, output_dir):
    fig, ax = plt.subplots(figsize=(14, 3))

    energies = []
    midpoints = []
    for seg in segments:
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        chunk = y[start_sample:end_sample]
        if len(chunk) > 0:
            energies.append(float(np.sqrt(np.mean(chunk ** 2))))
            midpoints.append((seg["start"] + seg["end"]) / 2)

    ax.bar(midpoints, energies, width=0.4, color='coral', alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMS Energy")
    ax.set_title("Segment Energy over Time")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "segment_energy.png"), dpi=150)
    plt.close()
