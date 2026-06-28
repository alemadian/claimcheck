"""
Skeleton: a real model-backed judge.

Subclass JudgeClient, call your provider at temperature 0 with a fixed seed, and
return the strict JSON the rubric asks for. Pin the model id and prompt version
in JudgeConfig so every verdict records exactly which judge produced it; that is
what makes the calibration and drift signals meaningful over time.

Point the CLI at it:  claimcheck run --judge examples.real_judge_adapter:make_judge ...
"""

from __future__ import annotations

import json

from claimcheck.judges.client import JudgeClient, JudgeConfig


class RealJudge(JudgeClient):
    def __init__(self) -> None:
        # Bump prompt_version whenever the rubric text changes; the fingerprint
        # recorded in each verdict will change with it.
        self.config = JudgeConfig(
            model_id="your-judge-model-id",
            temperature=0.0,
            seed=7,
            prompt_version="v1",
        )

    def complete(self, system: str, user: str) -> str:
        # raw = your_client.chat(
        #     model=self.config.model_id,
        #     temperature=self.config.temperature,
        #     seed=self.config.seed,
        #     messages=[{"role": "system", "content": system},
        #               {"role": "user", "content": user}],
        # )
        # The harness expects a JSON object: {"score": 0|1|2,
        # "supporting_span": "...", "justification": "..."}.
        # return raw
        raise NotImplementedError("Wire your provider call here.")


def make_judge() -> JudgeClient:
    return RealJudge()
