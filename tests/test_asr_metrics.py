import unittest

from lifemate.evaluation.asr_metrics import (
    character_error_rate,
    compute_asr_metrics,
    entity_error_rate,
    word_error_rate,
)
from lifemate.evaluation.benchmark import validation_split
from lifemate.evaluation.benchmark import benchmark_models
from lifemate.evaluation.error_analysis import analyze_errors


class ASRMetricTests(unittest.TestCase):
    def test_identical_text_has_zero_errors(self):
        self.assertEqual(word_error_rate("Xin chào!", "xin chào"), 0.0)
        self.assertEqual(character_error_rate("Xin chào!", "xin chào"), 0.0)

    def test_word_and_character_error_rates(self):
        self.assertAlmostEqual(word_error_rate("xin chào bạn", "xin bạn"), 1 / 3)
        self.assertGreater(character_error_rate("xin chào", "xin chao"), 0)

    def test_entity_errors_detect_different_times(self):
        self.assertEqual(entity_error_rate("hẹn lúc 09:00", "hẹn lúc 10:00"), 1.0)

    def test_aggregate_metrics_reject_mismatched_lists(self):
        with self.assertRaises(ValueError):
            compute_asr_metrics(["xin chào"], [])

    def test_validation_split_is_stable_and_disjoint_by_id(self):
        rows = [{"id": f"sample-{index}"} for index in range(100)]
        first = validation_split(rows, 0.2, seed=7)
        second = validation_split(rows, 0.2, seed=7)
        self.assertEqual([row["id"] for row in first], [row["id"] for row in second])
        self.assertEqual(len(first), 20)
        self.assertEqual(len({row["id"] for row in first}), 20)

    def test_error_analysis_splits_groups(self):
        result = analyze_errors(
            [
                {"id": "a", "group": "north", "reference": "xin chào", "hypothesis": "xin chào"},
                {"id": "b", "group": "south", "reference": "chào bạn", "hypothesis": "chào"},
            ]
        )
        self.assertEqual(set(result["groups"]), {"north", "south"})
        self.assertEqual(result["error_types"], {"deletion": 1})

    def test_models_share_samples_and_report_city_as_proxy(self):
        class FakeModel:
            def transcribe_batch(self, audio_paths, contexts=None):
                return ["xin chào" for _ in audio_paths]

        samples = [
            {
                "id": "one",
                "audio": "one.wav",
                "audio_text": "xin chào",
                "context": {"user_profile": {"city": "Huế"}},
            },
            {
                "id": "two",
                "audio": "two.wav",
                "audio_text": "chào bạn",
                "context": {"user_profile": {"city": "Cần Thơ"}},
            },
        ]
        result = benchmark_models(
            {"model-a": FakeModel(), "model-b": FakeModel()},
            samples,
            ".",
            batch_size=2,
        )
        self.assertEqual(result["validation_ids"], ["one", "two"])
        self.assertEqual(
            result["grouping_basis"],
            ["city_proxy_not_accent"],
        )
        self.assertEqual(result["models"]["model-a"]["metrics"]["samples"], 2)


if __name__ == "__main__":
    unittest.main()
