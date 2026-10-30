import unittest

from lifemate.inference.pipeline import InferencePipeline
from lifemate.speech.pipeline import SpeechPipeline


class FakeSpeech:
    def transcribe(self, audio, context=None, history=None):
        return "bật đèn"


class FakeEnvironment:
    def __init__(self):
        self.calls = []

    def call(self, name, arguments):
        self.calls.append({"name": name, "arguments": arguments})
        return {"ok": True, "state": "updated"}

    def get_tool_calls(self):
        return self.calls

    def get_state_hash(self):
        return "state-hash"


class FakeAgent:
    def __init__(self):
        self.rounds = 0

    def plan(self, transcript, context, history, tool_results):
        self.rounds += 1
        if self.rounds == 1:
            return {"tool_calls": [{"name": "smart_home.get_status", "arguments": {}}]}
        if self.rounds == 2:
            self.seen_results = tool_results
            return {"tool_calls": [{"name": "smart_home.light_set", "arguments": {"power": "off"}}]}
        return {"tool_calls": [], "final_response": "Đã tắt đèn"}


class IntegrationTests(unittest.TestCase):
    def test_speech_agent_tools_loop_and_submission_fields(self):
        environment = FakeEnvironment()
        agent = FakeAgent()
        pipeline = InferencePipeline(SpeechPipeline(FakeSpeech()))
        result = pipeline.run(
            audio="request.wav",
            context={"current_time": "2026-11-28"},
            history=[],
            agent=agent,
            tool_env=environment,
            sample_id="sample-1",
        )
        self.assertEqual(result["id"], "sample-1")
        self.assertEqual(result["transcript"], "bật đèn")
        self.assertEqual(
            [call["name"] for call in environment.calls],
            ["smart_home.get_status", "smart_home.light_set"],
        )
        self.assertEqual(len(agent.seen_results), 1)
        self.assertEqual(result["state_hash"], "state-hash")
        self.assertEqual(result["final_response"], "Đã tắt đèn")

    def test_tool_call_limit_is_reported(self):
        class EndlessAgent:
            def plan(self, **kwargs):
                return [{"name": "test.tool", "arguments": {}}]

        with self.assertRaisesRegex(RuntimeError, "limit"):
            InferencePipeline(SpeechPipeline(FakeSpeech()), max_tool_calls=1).run(
                audio="request.wav",
                context={},
                history=[],
                agent=EndlessAgent(),
                tool_env=FakeEnvironment(),
            )

    def test_tool_metrics_account_for_duplicate_calls(self):
        from lifemate.evaluation.agent_metrics import tool_call_metrics

        call = {"name": "calendar.list_events", "arguments": {}}
        metrics = tool_call_metrics([call, call], [call])
        self.assertEqual(metrics["matched_calls"], 1)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()