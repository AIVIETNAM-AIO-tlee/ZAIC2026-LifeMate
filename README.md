# LifeMate — Vietnamese speech and tool-use assistant

This repository provides audio preprocessing, configurable ASR inference, Vietnamese transcript normalization, dataset analysis, ASR evaluation, and an adapter that connects speech recognition to an independently implemented agent/tool runtime.

## Setup

```powershell
python -m pip install -e ".[asr]"
```

The default ASR checkpoint is `openai/whisper-small`; the model cache is `models/asr`. Review `configs/asr.yaml` before downloading or running inference.

## Useful commands

```powershell
python scripts\analyze_audio.py
python scripts\evaluate_asr.py
python scripts\benchmark_runtime.py --limit 32
python -m unittest discover -s tests -v
```

Submission inference uses a planner factory provided by the separately scoped agent tasks:

```powershell
python scripts\run_inference.py --split public_test --agent my_agent:build_agent
```

See [architecture](docs/architecture.md), [evaluation](docs/evaluation.md), and [deployment](docs/deployment.md) for pipeline contracts and known dataset limitations.