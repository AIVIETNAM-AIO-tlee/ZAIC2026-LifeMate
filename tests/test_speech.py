from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from lifemate.speech.normalizer import normalize_text
from lifemate.speech.pipeline import SpeechPipeline
from lifemate.speech.preprocessing import load_wav


class FakeTranscriber:
    def __init__(self, transcript: str = "  NGÀY 5 tháng 3 năm 2026, 7 giờ 30 sáng! ") -> None:
        self.transcript = transcript
        self.context = None

    def transcribe(self, audio, context=None):
        self.context = context
        return self.transcript


class SpeechTests(unittest.TestCase):
    def make_wav(self, path: Path, channels: int = 2, sample_rate: int = 8_000) -> None:
        samples = np.zeros((sample_rate, channels), dtype="<i2")
        samples[:, 0] = 16384
        if channels > 1:
            samples[:, 1] = -16384
        with wave.open(str(path), "wb") as target:
            target.setnchannels(channels)
            target.setsampwidth(2)
            target.setframerate(sample_rate)
            target.writeframes(samples.tobytes())

    def test_reads_wav_mixes_stereo_and_resamples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            self.make_wav(path)
            audio = load_wav(path, target_sample_rate=16_000, mono=True)
        self.assertEqual(audio.sample_rate, 16_000)
        self.assertEqual(audio.samples.shape, (16_000,))
        self.assertTrue(np.allclose(audio.samples, 0.0, atol=1e-4))

    def test_normalizes_dates_times_and_currency(self):
        self.assertEqual(
            normalize_text("Ngày 5 tháng 3 năm 2026, 7 giờ 30 sáng; 50 nghìn đồng!"),
            "05/03/2026, 07:30; 50000 đồng!",
        )

    def test_preserves_address_device_names_and_code_switching(self):
        transcript = "  Địa chỉ 88 Nguyễn Văn Linh, đèn phòng ngủ nhỏ bật Spotify nhé. "
        self.assertEqual(
            normalize_text(transcript),
            "địa chỉ 88 nguyễn văn linh, đèn phòng ngủ nhỏ bật spotify nhé.",
        )

    def test_pipeline_forwards_user_profile_as_hotwords_and_normalizes_output(self):
        model = FakeTranscriber()
        pipeline = SpeechPipeline(model)
        result = pipeline.transcribe(
            "audio.wav",
            context={"user_profile": {"name": "Em Linh", "city": "Đà Nẵng"}},
        )
        self.assertEqual(result, "05/03/2026, 07:30!")
        self.assertEqual(model.context["hotwords"], ["Em Linh", "Đà Nẵng"])

    def test_transcribe_with_optional_intent_slot_extractor(self):
        model = FakeTranscriber("Bật đèn phòng khách")
        pipeline = SpeechPipeline(model)
        result = pipeline.transcribe_with_slots(
            "audio.wav",
            slot_extractor=lambda text, context: {"intent": "smart_home.light_set", "device": "phòng khách"},
        )
        self.assertEqual(result["intent_slots"]["intent"], "smart_home.light_set")
        self.assertEqual(result["transcript"], "bật đèn phòng khách")


if __name__ == "__main__":
    unittest.main()