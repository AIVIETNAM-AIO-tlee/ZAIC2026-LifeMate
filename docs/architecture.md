# LifeMate speech and inference architecture

## Speech path

`SpeechPipeline` accepts a WAV path, builds conservative ASR hints from user profile/history, calls an injected transcriber, then normalizes Vietnamese dates, times, currency expressions, whitespace, and punctuation. `ASRModel` loads a configured Hugging Face Whisper or CTC checkpoint; WAV decoding, channel mixing, and sample-rate conversion are implemented in `speech/preprocessing.py`.

Intent/slot extraction is an optional adapter (`transcribe_with_slots`). It does not fabricate slots when no extractor is configured; connect the later agent implementation or another approved extractor through the callable interface.

## Evaluation path

`scripts/analyze_audio.py` summarizes the labeled split. `evaluation/benchmark.py` creates one deterministic held-out ID set and evaluates every configured ASR model on those same samples. Metrics and per-group error analysis are implemented without requiring an external scoring service. Group labels prefer explicit accent/dialect/region metadata; city is only reported as a location proxy.

## Inference path

`InferencePipeline` connects an ASR transcriber to an injected agent planner and a single sample-scoped `ToolEnv`. The planner sees prior tool results and can request additional rounds. Tool calls are executed in sequence; malformed plans, agent errors, and call-limit overruns are raised explicitly. The CLI requires `--agent module:factory`, because the agent planner tasks on the separate “Tasks-Thành” sheet are intentionally not implemented here.