from crg_ce.estimators.openhands.config import AgentConfig, EntailmenntPromptConfig
from crg_ce.evaluators.contextual_entailment import ContextualEntailmentEvaluator, EntailmentAssessment


def test_contextual_entailment_loads_configured_examples_and_uses_tenacious_completion(monkeypatch) -> None:
    # This verifies examples loaded from a configured package resource and other optional inputs reach the task prompt.
    # It assumes structured completion is the only LLM boundary for contextual entailment.
    calls = []

    def fake_complete(**kwargs):
        calls.append(kwargs)
        return EntailmentAssessment(label="entailment", rationale="The two child claims suffice.")

    monkeypatch.setattr("crg_ce.evaluators.contextual_entailment.tenaciously_complete_structured", fake_complete)
    evaluator = ContextualEntailmentEvaluator(
        AgentConfig(model_name="test/evaluator"),
        EntailmenntPromptConfig(examples="prompts/gqa/local/entailment/v0/examples.txt"),  # type: ignore
    )

    result = evaluator.evaluate(
        "The tests pass AND the patch is applied.",
        "The fix is complete.",
        agent_task_input="Fix the regression.",
        trajectory="Ran tests.",
    )

    assert result.label == "entailment"
    assert calls[0]["output_model"] is EntailmentAssessment
    assert calls[0]["messages"][0]["content"] == evaluator.system_prompt.render()
    prompt = calls[0]["messages"][1]["content"]
    assert "<examples>" in prompt
    assert "<agent_task_input>\nFix the regression." in prompt
    assert "<trajectory>\nRan tests." in prompt
    assert "<premise>\nThe tests pass AND the patch is applied." in prompt
