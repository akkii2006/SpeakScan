import numpy as np
import librosa


def estimate_noise_floor(y, sr, frame_length=2048, hop_length=512, percentile=10):
    energies = []
    for i in range(0, len(y) - frame_length, hop_length):
        frame = y[i:i + frame_length]
        energies.append(np.mean(frame ** 2))

    if not energies:
        return 1e-10

    # treat the quietest portions of the audio as an estimate of the noise floor
    noise_power = np.percentile(energies, percentile)
    return max(noise_power, 1e-10)


def compute_segment_snr(y, sr, start, end, noise_power):
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    chunk = y[start_sample:end_sample]

    if len(chunk) == 0:
        return 0.0

    signal_power = np.mean(chunk ** 2)
    if signal_power <= 0:
        return 0.0

    snr_db = 10 * np.log10(signal_power / noise_power)
    return round(float(snr_db), 2)


def detect_clipping(y, sr, start, end, threshold=0.99, max_ratio=0.001):
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    chunk = y[start_sample:end_sample]

    if len(chunk) == 0:
        return False

    clipped = np.sum(np.abs(chunk) >= threshold)
    ratio = clipped / len(chunk)
    return ratio > max_ratio


def detect_overlap(start, end, diar_segments, speaker, min_overlap_ratio=0.3):
    seg_duration = end - start
    if seg_duration <= 0:
        return False

    overlap_time = 0.0
    for d_seg in diar_segments:
        if d_seg["speaker"] == speaker:
            continue

        overlap_start = max(start, d_seg["start"])
        overlap_end = min(end, d_seg["end"])
        overlap_time += max(0.0, overlap_end - overlap_start)

    return (overlap_time / seg_duration) > min_overlap_ratio


def evaluate_segments(audio_path, segments, diar_segments=None,
                       min_snr_db=5.0, clipping_threshold=0.99,
                       clipping_max_ratio=0.001, overlap_ratio=0.3,
                       min_duration=0.1):
    y, sr = librosa.load(audio_path, sr=None)
    noise_power = estimate_noise_floor(y, sr)

    evaluated = []
    for seg in segments:
        duration = seg["end"] - seg["start"]
        flags = []

        if duration < min_duration:
            flags.append("too_short")

        snr_db = compute_segment_snr(y, sr, seg["start"], seg["end"], noise_power)
        if snr_db < min_snr_db:
            flags.append("low_snr")

        if detect_clipping(y, sr, seg["start"], seg["end"], clipping_threshold, clipping_max_ratio):
            flags.append("clipping")

        if diar_segments is not None:
            speaker = seg.get("speaker", "UNKNOWN")
            if detect_overlap(seg["start"], seg["end"], diar_segments, speaker, overlap_ratio):
                flags.append("overlapping_speech")

        evaluated.append({
            **seg,
            "snr_db": snr_db,
            "quality_flags": flags,
            "quality_ok": len(flags) == 0
        })

    return evaluated