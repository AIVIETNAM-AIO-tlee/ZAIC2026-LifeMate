import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_audio import analyze


class AudioEDATests(unittest.TestCase):
    def test_audio_transcript_and_region_metadata_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "train"
            audio_dir = root / "audio" / "sample-1"
            audio_dir.mkdir(parents=True)
            audio_path = audio_dir / "sample-1.wav"
            with wave.open(str(audio_path), "wb") as target:
                target.setnchannels(1)
                target.setsampwidth(2)
                target.setframerate(8_000)
                target.writeframes(np.zeros(8_000, dtype="<i2").tobytes())
            sample = {
                "id": "sample-1",
                "audio": "audio/sample-1/sample-1.wav",
                "audio_text": "Xin chào LifeMate",
                "context": {
                    "user_profile": {"city": "Huế"},
                },
            }
            with (root / "train.jsonl").open("w", encoding="utf-8") as target:
                target.write(json.dumps(sample, ensure_ascii=False) + "\n")
            report = analyze(root)
        self.assertEqual(report["audio"]["sample_rates"], {8_000: 1})
        self.assertEqual(report["audio"]["channel_counts"], {1: 1})
        self.assertEqual(report["transcripts"]["with_transcript"], 1)
        self.assertEqual(report["metadata"]["region_or_accent_available"], 0)
        self.assertEqual(report["metadata"]["city_counts"], {"Huế": 1})


if __name__ == "__main__":
    unittest.main()
