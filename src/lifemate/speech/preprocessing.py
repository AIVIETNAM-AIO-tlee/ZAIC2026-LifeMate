"""Audio loading and preparation for speech recognition."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import wave

import numpy as np


@dataclass(frozen=True)
class AudioData:
    samples: np.ndarray
    sample_rate: int


def _decode_pcm(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = (
            packed[:, 0].astype(np.int32)
            | (packed[:, 1].astype(np.int32) << 8)
            | (packed[:, 2].astype(np.int32) << 16)
        )
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float32) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    raise ValueError(f"Unsupported PCM sample width: {sample_width} bytes")


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples.astype(np.float32, copy=False)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive")
    try:
        from scipy.signal import resample_poly
    except ImportError:
        output_length = max(1, round(len(samples) * target_rate / source_rate))
        source_positions = np.arange(len(samples), dtype=np.float64)
        target_positions = np.arange(output_length, dtype=np.float64) * source_rate / target_rate
        return np.interp(target_positions, source_positions, samples).astype(np.float32)

    from math import gcd

    divisor = gcd(source_rate, target_rate)
    return resample_poly(samples, target_rate // divisor, source_rate // divisor).astype(np.float32)


def load_wav(
    path: str | Path,
    target_sample_rate: int | None = 16_000,
    mono: bool = True,
) -> AudioData:
    """Read an uncompressed PCM WAV, optionally convert it to mono and resample it."""
    with wave.open(str(path), "rb") as wav:
        if wav.getcomptype() != "NONE":
            raise ValueError(f"Compressed WAV is not supported: {path}")
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        samples = _decode_pcm(wav.readframes(wav.getnframes()), wav.getsampwidth())

    if channels < 1:
        raise ValueError(f"WAV has an invalid channel count: {path}")
    if sample_rate <= 0 or samples.size == 0:
        raise ValueError(f"WAV has no samples or an invalid sample rate: {path}")
    samples = samples.reshape(-1, channels)
    if mono:
        samples = samples.mean(axis=1, dtype=np.float32)
    if target_sample_rate is not None and sample_rate != target_sample_rate:
        samples = _resample(samples, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    if mono or channels == 1:
        samples = samples.reshape(-1)
    return AudioData(samples=np.asarray(samples, dtype=np.float32), sample_rate=sample_rate)


def prepare_audio(
    path: str | Path,
    target_sample_rate: int = 16_000,
    mono: bool = True,
) -> AudioData:
    """Compatibility-friendly name for loading model-ready audio."""
    return load_wav(path, target_sample_rate=target_sample_rate, mono=mono)