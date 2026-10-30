# Local inference setup

Install the base project and ASR dependencies:

```powershell
python -m pip install -e ".[asr]"
```

The default model is `openai/whisper-small`. On first use, Transformers downloads it into the configured `models/asr` cache; set `local_files_only: true` and provide a compatible local checkpoint for offline operation. Choose CPU/CUDA and precision through `configs/asr.yaml`.

Submission generation requires a project-specific agent factory implementing `plan(transcript, context, history, tool_results)`. The factory is supplied by the separately scoped agent work:

```powershell
python scripts\run_inference.py --split public_test --data-root data --runtime-config configs\runtime.yaml --agent my_agent:build_agent
```

The CLI creates one tool environment per sample and writes only the required `id`, `tool_calls`, and `state_hash` fields to `submission.jsonl`. It does not provide an empty-call fallback when the agent is unavailable.