import os
import json
import html


SPEAKER_COLORS = [
    "#4361ee", "#f72585", "#4cc9f0", "#80ed99",
    "#ffb703", "#9d4edd", "#fb5607", "#06d6a0"
]


STYLE = """
<style>
* { box-sizing: border-box; }
body {
    font-family: -apple-system, Segoe UI, Roboto, sans-serif;
    background: #f4f5f7;
    color: #1a1a1a;
    margin: 0;
    padding: 2rem;
}
.container { max-width: 900px; margin: 0 auto; }
h1 { font-size: 1.8rem; margin-bottom: 0.25rem; }
.subtitle { color: #666; margin-bottom: 1.5rem; }
.card {
    background: #fff;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.card h2 { margin-top: 0; font-size: 1.1rem; }
.overview-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 1rem;
}
.overview-grid > div { display: flex; flex-direction: column; }
.label { font-size: 0.8rem; color: #888; }
.value { font-size: 1.4rem; font-weight: 600; }
table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid #eee; font-size: 0.9rem; }
.tags { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.tag {
    background: #eef2ff;
    color: #4361ee;
    border-radius: 999px;
    padding: 0.2rem 0.7rem;
    font-size: 0.85rem;
}
.image-block { margin-bottom: 1rem; }
.image-block img { width: 100%; border-radius: 8px; }
.image-block h3 { font-size: 0.95rem; margin-bottom: 0.4rem; color: #555; }
.segment {
    padding: 0.6rem 0;
    border-bottom: 1px solid #f0f0f0;
}
.segment:last-child { border-bottom: none; }
.timestamp { font-size: 0.8rem; color: #999; margin-right: 0.6rem; font-family: monospace; }
.speaker {
    color: #fff;
    border-radius: 6px;
    padding: 0.1rem 0.5rem;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 0.5rem;
}
.emotion {
    font-size: 0.8rem;
    color: #888;
    text-transform: capitalize;
    margin-right: 0.5rem;
}
.lang-badge {
    font-size: 0.75rem;
    background: #f0fdf4;
    color: #16a34a;
    border-radius: 4px;
    padding: 0.1rem 0.4rem;
    font-family: monospace;
    margin-right: 0.5rem;
}
.text { margin: 0.3rem 0 0 0; line-height: 1.5; }
</style>
"""


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def speaker_color_map(segments):
    speakers = sorted({seg.get("speaker", "UNKNOWN") for seg in segments})
    return {sp: SPEAKER_COLORS[i % len(SPEAKER_COLORS)] for i, sp in enumerate(speakers)}


def render_stats_section(stats):
    if not stats:
        return ""

    rows = ""
    for speaker, data in stats.get("speaker_stats", {}).items():
        rows += f"""
        <tr>
            <td>{html.escape(speaker)}</td>
            <td>{data['talk_time_seconds']}s</td>
            <td>{data['talk_time_percent']}%</td>
            <td>{data['turn_count']}</td>
        </tr>"""

    emotion_rows = ""
    for emotion, data in stats.get("emotion_distribution", {}).items():
        emotion_rows += f"""
        <tr>
            <td>{html.escape(emotion)}</td>
            <td>{data['count']}</td>
            <td>{data['percent']}%</td>
        </tr>"""

    lang_rows = ""
    for lang, data in stats.get("language_distribution", {}).items():
        lang_rows += f"""
        <tr>
            <td>{html.escape(lang)}</td>
            <td>{data['count']}</td>
            <td>{data['percent']}%</td>
        </tr>"""

    lang_section = f"""
    <section class="card">
        <h2>Language Distribution</h2>
        <table>
            <tr><th>Language</th><th>Segments</th><th>Share</th></tr>
            {lang_rows}
        </table>
    </section>
    """ if lang_rows else ""

    return f"""
    <section class="card">
        <h2>Overview</h2>
        <div class="overview-grid">
            <div><span class="label">Duration</span><span class="value">{stats['total_duration_hours']:.2f} hrs</span></div>
            <div><span class="label">Segments</span><span class="value">{stats['num_segments']}</span></div>
            <div><span class="label">Speakers</span><span class="value">{stats['num_speakers']}</span></div>
            <div><span class="label">Avg Segment</span><span class="value">{stats['avg_segment_length_seconds']}s</span></div>
        </div>
    </section>
    <section class="card">
        <h2>Speaker Talk Time</h2>
        <table>
            <tr><th>Speaker</th><th>Talk Time</th><th>Share</th><th>Turns</th></tr>
            {rows}
        </table>
    </section>
    <section class="card">
        <h2>Emotion Distribution</h2>
        <table>
            <tr><th>Emotion</th><th>Count</th><th>Share</th></tr>
            {emotion_rows}
        </table>
    </section>
    {lang_section}
    """


def render_keywords_section(keywords):
    if not keywords:
        return ""

    tags = "".join(f'<span class="tag">{html.escape(kw["keyword"])}</span>' for kw in keywords)

    return f"""
    <section class="card">
        <h2>Keywords</h2>
        <div class="tags">{tags}</div>
    </section>
    """


def render_images_section(video_dir):
    images = [
        ("waveform.png", "Waveform"),
        ("mel_spectrogram.png", "Mel Spectrogram"),
        ("segment_energy.png", "Segment Energy"),
        ("emotion_timeline.png", "Emotion Timeline")
    ]

    blocks = ""
    for filename, label in images:
        if os.path.exists(os.path.join(video_dir, filename)):
            blocks += f"""
            <div class="image-block">
                <h3>{label}</h3>
                <img src="{filename}" alt="{label}">
            </div>"""

    if not blocks:
        return ""

    return f"""
    <section class="card">
        <h2>Visualizations</h2>
        {blocks}
    </section>
    """


def render_transcript_section(segments, colors):
    rows = ""
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        color = colors.get(speaker, "#999999")
        emotion = seg.get("emotion", "neutral")
        language = seg.get("language", "")
        lang_badge = f'<span class="lang-badge">{html.escape(language)}</span>' if language and language != "unknown" else ""
        rows += f"""
        <div class="segment">
            <span class="timestamp">{seg['start']:.2f}s - {seg['end']:.2f}s</span>
            <span class="speaker" style="background-color: {color}">{html.escape(speaker)}</span>
            <span class="emotion">{html.escape(emotion)}</span>
            {lang_badge}
            <p class="text">{html.escape(seg['text'])}</p>
        </div>"""

    return f"""
    <section class="card">
        <h2>Transcript</h2>
        {rows}
    </section>
    """


def generate_report(video_dir, title):
    segments = load_json(os.path.join(video_dir, "results.json"))
    if segments is None:
        raise FileNotFoundError(f"results.json not found in {video_dir}")

    stats = load_json(os.path.join(video_dir, "stats.json"))
    keywords = load_json(os.path.join(video_dir, "keywords.json"))
    colors = speaker_color_map(segments)

    safe_title = html.escape(title)

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{safe_title} - SpeakScan Report</title>
{STYLE}
</head>
<body>
<div class="container">
    <h1>{safe_title}</h1>
    <p class="subtitle">SpeakScan analysis report</p>
    {render_stats_section(stats)}
    {render_images_section(video_dir)}
    {render_keywords_section(keywords)}
    {render_transcript_section(segments, colors)}
</div>
</body>
</html>"""

    output_path = os.path.join(video_dir, "report.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(page)

    return output_path
