# ASR and integration evaluation

## Audio EDA

From the repository root, run:

```powershell
python scripts\analyze_audio.py --data-root data\train --output-dir experiments\reports
```

The script writes a Markdown summary and a per-sample JSON report with sample rate, channel count, duration, transcript lengths/text, city, and any explicit region/accent metadata. If the dataset has no accent labels, that absence is recorded rather than inferred from city.

## ASR benchmark

Configure one or more entries in `configs/asr.yaml`, then run:

```powershell
python scripts\evaluate_asr.py --data-root data\train --runtime-config configs\runtime.yaml
```

The benchmark uses a seeded, deterministic validation split and reuses precisely the same sample IDs for each model. It reports corpus WER, CER, entity error rate, per-region/group metrics, elapsed time, and word-level error categories in `experiments\asr\benchmark.json`. To reproduce the split, retain the reported fraction and seed.

WER and CER are computed on normalized, case-insensitive Vietnamese text. Entity error rate compares detected dates, times, currency amounts, and phone numbers; it complements rather than replaces WER/CER.

## Runtime measurement

```powershell
python scripts\benchmark_runtime.py --data-root data\train --runtime-config configs\runtime.yaml --limit 32 --batch-size 1
```

The JSON output records model, device, precision, batch size, latency, and throughput. Use the same audio subset when comparing runtime settings.

## Known dataset limitation

The included training JSONL has transcripts and user-profile city, but no dedicated accent, dialect, or speaker-region annotation. Current per-group scores use explicit labels when available and otherwise group on city as an explicitly named location proxy, not as evidence of accent.