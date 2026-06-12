import os
import json
import yake


def extract_keywords(text, max_keywords=10, language="en", max_ngram_size=2):
    if not text or not text.strip():
        return []

    extractor = yake.KeywordExtractor(
        lan=language,
        n=max_ngram_size,
        top=max_keywords
    )

    # YAKE scores are "lower is better" - lower score means more relevant keyword
    keywords = extractor.extract_keywords(text)

    return [
        {"keyword": kw, "score": round(score, 4)}
        for kw, score in keywords
    ]


def generate_keywords_report(transcript, video_dir, max_keywords=10):
    keywords = extract_keywords(transcript, max_keywords=max_keywords)

    output_path = os.path.join(video_dir, "keywords.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(keywords, f, indent=2, ensure_ascii=False)

    return keywords, output_path