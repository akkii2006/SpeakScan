import json
import os


def load_segments(video_dir):
    path = os.path.join(video_dir, "results.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_speaker_stats(segments):
    speaker_time = {}
    speaker_turns = {}
    prev_speaker = None

    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        duration = max(0.0, seg["end"] - seg["start"])
        speaker_time[speaker] = speaker_time.get(speaker, 0.0) + duration

        if speaker != prev_speaker:
            speaker_turns[speaker] = speaker_turns.get(speaker, 0) + 1
            prev_speaker = speaker

    total_time = sum(speaker_time.values())

    stats = {}
    for speaker, time_s in speaker_time.items():
        stats[speaker] = {
            "talk_time_seconds": round(time_s, 2),
            "talk_time_percent": round((time_s / total_time * 100) if total_time > 0 else 0.0, 2),
            "turn_count": speaker_turns.get(speaker, 0)
        }

    return stats


def compute_emotion_distribution(segments):
    counts = {}
    for seg in segments:
        emotion = seg.get("emotion", "neutral")
        counts[emotion] = counts.get(emotion, 0) + 1

    total = sum(counts.values())
    distribution = {}
    for emotion, count in counts.items():
        distribution[emotion] = {
            "count": count,
            "percent": round((count / total * 100) if total > 0 else 0.0, 2)
        }
    return distribution


def compute_language_distribution(segments):
    counts = {}
    for seg in segments:
        lang = seg.get("language", "unknown")
        counts[lang] = counts.get(lang, 0) + 1

    total = sum(counts.values())
    return {
        lang: {
            "count": count,
            "percent": round((count / total * 100) if total > 0 else 0.0, 2)
        }
        for lang, count in counts.items()
    }


def compute_emotion_by_speaker(segments):
    speaker_emotions = {}
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        emotion = seg.get("emotion", "neutral")
        speaker_emotions.setdefault(speaker, {})
        speaker_emotions[speaker][emotion] = speaker_emotions[speaker].get(emotion, 0) + 1

    result = {}
    for speaker, counts in speaker_emotions.items():
        total = sum(counts.values())
        result[speaker] = {
            emotion: round((count / total * 100) if total > 0 else 0.0, 2)
            for emotion, count in counts.items()
        }
    return result


def compute_emotion_timeline(segments):
    timeline = []
    for seg in segments:
        timeline.append({
            "start": seg["start"],
            "end": seg["end"],
            "speaker": seg.get("speaker", "UNKNOWN"),
            "emotion": seg.get("emotion", "neutral"),
            "emotion_score": seg.get("emotion_score", 0.0)
        })
    return timeline


def compute_dataset_stats(segments):
    if not segments:
        return {
            "total_duration_seconds": 0.0,
            "total_duration_hours": 0.0,
            "num_segments": 0,
            "avg_segment_length_seconds": 0.0,
            "num_speakers": 0,
            "emotion_distribution": {},
            "language_distribution": {},
            "speaker_stats": {},
            "emotion_by_speaker": {}
        }

    durations = [max(0.0, seg["end"] - seg["start"]) for seg in segments]
    total_duration = sum(durations)
    speakers = {seg.get("speaker", "UNKNOWN") for seg in segments}

    return {
        "total_duration_seconds": round(total_duration, 2),
        "total_duration_hours": round(total_duration / 3600, 4),
        "num_segments": len(segments),
        "avg_segment_length_seconds": round(total_duration / len(segments), 2),
        "num_speakers": len(speakers),
        "emotion_distribution": compute_emotion_distribution(segments),
        "language_distribution": compute_language_distribution(segments),
        "speaker_stats": compute_speaker_stats(segments),
        "emotion_by_speaker": compute_emotion_by_speaker(segments)
    }


def generate_stats_report(video_dir):
    segments = load_segments(video_dir)

    stats = compute_dataset_stats(segments)
    stats["emotion_timeline"] = compute_emotion_timeline(segments)

    output_path = os.path.join(video_dir, "stats.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    return stats, output_path
