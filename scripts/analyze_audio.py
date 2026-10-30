"""Create a transcript/audio/metadata EDA report for a LifeMate JSONL split."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import wave


_REGION_KEYS = ("accent", "dialect", "region", "speaker_region", "province")


def _read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _metadata_region(item: dict) -> str | None:
    context = item.get("context", {})
    profile = context.get("user_profile", {}) if isinstance(context, dict) else {}
    for candidate in (item, context, profile):
        if isinstance(candidate, dict):
            for key in _REGION_KEYS:
                value = candidate.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return None


def analyze(data_root: Path) -> dict:
    split = data_root.name
    dataset_file = data_root / f"{split}.jsonl"
    if not dataset_file.is_file():
        raise FileNotFoundError(f"Dataset JSONL not found: {dataset_file}")
    rows = _read_rows(dataset_file)
    if not rows:
        raise ValueError(f"Dataset contains no samples: {dataset_file}")

    samples = []
    rates, channels, durations, transcript_words, transcript_chars = [], [], [], [], []
    city_counts, region_counts = Counter(), Counter()
    for item in rows:
        audio_path = data_root / item["audio"]
        with wave.open(str(audio_path), "rb") as wav:
            sample_rate = wav.getframerate()
            channel_count = wav.getnchannels()
            duration = wav.getnframes() / sample_rate
        transcript = item.get("audio_text")
        context = item.get("context", {})
        profile = context.get("user_profile", {}) if isinstance(context, dict) else {}
        city = profile.get("city") if isinstance(profile, dict) else None
        region = _metadata_region(item)
        if isinstance(city, str) and city.strip():
            city_counts[city.strip()] += 1
        region_counts[region or "unavailable"] += 1
        rates.append(sample_rate)
        channels.append(channel_count)
        durations.append(duration)
        if isinstance(transcript, str):
            transcript_words.append(len(transcript.split()))
            transcript_chars.append(len(transcript))
        samples.append(
            {
                "id": item["id"],
                "audio": item["audio"],
                "sample_rate": sample_rate,
                "channels": channel_count,
                "duration_seconds": round(duration, 4),
                "transcript": transcript,
                "transcript_words": len(transcript.split()) if isinstance(transcript, str) else None,
                "transcript_characters": len(transcript) if isinstance(transcript, str) else None,
                "city": city,
                "region_or_accent": region,
            }
        )

    return {
        "split": split,
        "sample_count": len(samples),
        "audio": {
            "sample_rates": dict(sorted(Counter(rates).items())),
            "channel_counts": dict(sorted(Counter(channels).items())),
            "duration_seconds": {
                "min": round(min(durations), 4),
                "median": round(statistics.median(durations), 4),
                "mean": round(statistics.mean(durations), 4),
                "max": round(max(durations), 4),
            },
        },
        "transcripts": {
            "with_transcript": len(transcript_words),
            "word_count": {
                "min": min(transcript_words) if transcript_words else None,
                "median": statistics.median(transcript_words) if transcript_words else None,
                "mean": statistics.mean(transcript_words) if transcript_words else None,
                "max": max(transcript_words) if transcript_words else None,
            },
            "character_count": {
                "min": min(transcript_chars) if transcript_chars else None,
                "median": statistics.median(transcript_chars) if transcript_chars else None,
                "mean": statistics.mean(transcript_chars) if transcript_chars else None,
                "max": max(transcript_chars) if transcript_chars else None,
            },
        },
        "metadata": {
            "region_or_accent_counts": dict(sorted(region_counts.items())),
            "city_counts": dict(sorted(city_counts.items())),
            "region_or_accent_available": sum(region_counts.values()) - region_counts.get("unavailable", 0),
            "city_is_location_proxy_not_accent": True,
        },
        "samples": samples,
    }


def _markdown(report: dict) -> str:
    audio = report["audio"]
    transcript = report["transcripts"]
    metadata = report["metadata"]
    lines = [
        f"# Audio EDA — {report['split']}",
        "",
        f"- Samples: {report['sample_count']}",
        f"- Sample rates (count): `{audio['sample_rates']}`",
        f"- Channel counts (count): `{audio['channel_counts']}`",
        f"- Duration seconds (min/median/mean/max): `{audio['duration_seconds']}`",
        f"- Transcript word count (min/median/mean/max): `{transcript['word_count']}`",
        f"- Region/accent metadata coverage: {metadata['region_or_accent_available']}/{report['sample_count']}",
        "",
        "## City metadata (location proxy only; not an accent label)",
        "",
    ]
    lines.extend(f"- {city}: {count}" for city, count in metadata["city_counts"].items())
    if not metadata["city_counts"]:
        lines.append("- No city metadata was available.")
    lines.extend(
        [
            "",
            "Region/accent counts: "
            + ", ".join(f"{key}: {value}" for key, value in metadata["region_or_accent_counts"].items()),
            "",
            "Full per-sample audio, transcript, and metadata records are in the adjacent JSON report.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/reports"))
    args = parser.parse_args()
    report = analyze(args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{report['split']}_audio_eda"
    with (args.output_dir / f"{stem}.json").open("w", encoding="utf-8") as target:
        json.dump(report, target, ensure_ascii=False, indent=2)
    (args.output_dir / f"{stem}.md").write_text(_markdown(report), encoding="utf-8")
    print(f"Analyzed {report['sample_count']} samples -> {args.output_dir / (stem + '.md')}")


if __name__ == "__main__":
    main()
