from __future__ import annotations

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, List, Optional

import dspy


class RSAggregate(dspy.Signature):
    """Aggregate and refine candidate solutions into one correct, high-quality answer."""

    instruction = dspy.InputField(desc="Aggregation instructions.")
    problem = dspy.InputField()
    candidates = dspy.InputField(desc="Candidate solutions, possibly empty.")
    answer = dspy.OutputField(desc="Final answer only.")


class RSAPopulation(dspy.Signature):
    """Solve the problem directly and return the final answer only."""

    instruction = dspy.InputField(desc="Answering instructions.")
    problem = dspy.InputField()
    answer = dspy.OutputField(desc="Final answer only.")


@dataclass(frozen=True)
class RSASample:
    reasoning: str
    answer: str


def _format_candidates(candidate_answers: Optional[List[RSASample]]) -> str:
    if not candidate_answers:
        return ""
    if len(candidate_answers) == 1:
        sample = candidate_answers[0]
        return (
            "Candidate solution (may contain mistakes):\n"
            "---- Candidate ----\n"
            f"Reasoning:\n{sample.reasoning.strip()}\n\n"
            f"Answer:\n{sample.answer.strip()}"
        )
    lines: List[str] = ["Candidate solutions (may contain mistakes):"]
    for i, sample in enumerate(candidate_answers, 1):
        lines.append(
            f"---- Solution {i} ----\n"
            f"Reasoning:\n{sample.reasoning.strip()}\n\n"
            f"Answer:\n{sample.answer.strip()}"
        )
    return "\n".join(lines)


def _call_predict(
    predictor: dspy.Module,
    *,
    instruction: str,
    problem: str,
    candidates: Optional[List[RSASample]],
    temperature: float,
    max_tokens: Optional[int],
    rollout_id: int,
) -> RSASample:
    kwargs = {"temperature": temperature, "rollout_id": rollout_id}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    result = predictor(
        instruction=instruction.strip(),
        problem=problem.strip(),
        candidates=_format_candidates(candidates),
        **kwargs,
    )
    return RSASample(
        reasoning=(getattr(result, "reasoning", "") or "").strip(),
        answer=(result.answer or "").strip(),
    )


def total_cost_usd(lm: dspy.LM) -> float:
    total = 0.0
    for entry in lm.history:
        cost = entry.get("cost", 0.0) or 0.0
        total += float(cost)
    return total


@dataclass
class RSAConfig:
    prompt: str
    model: str = "openai/gpt-4o-mini"
    population: int = 8
    subset: int = 3
    steps: int = 2
    temperature: float = 0.7
    population_temperature: Optional[float] = None
    aggregate_temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    rollout_base: Optional[int] = None
    debug: bool = False
    cache: bool = False
    parallel: int = 1
    progress_cb: Optional[Callable[[dict], None]] = None
    completion_cb: Optional[Callable[[dict], None]] = None


@dataclass
class RSAResult:
    answer: str
    population: List[RSASample]
    trace: List[dict]
    total_cost: float


def run_rsa(config: RSAConfig) -> RSAResult:
    if config.population < 1:
        raise ValueError("population must be >= 1")
    if config.subset < 1:
        raise ValueError("subset must be >= 1")
    if config.subset > config.population:
        raise ValueError("subset must be <= population")
    if config.steps < 0:
        raise ValueError("steps must be >= 0")

    if config.seed is not None:
        random.seed(config.seed)

    prompt = config.prompt.strip()
    if not prompt:
        raise ValueError("prompt is empty")

    if config.model.startswith("openai/") and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Export it in your environment.")

    lm = dspy.LM(config.model, cache=config.cache)
    dspy.configure(lm=lm)

    rollout_id = config.rollout_base or random.randint(1, 1_000_000)
    population_temperature = (
        config.population_temperature
        if config.population_temperature is not None
        else config.temperature
    )
    aggregate_temperature = (
        config.aggregate_temperature
        if config.aggregate_temperature is not None
        else config.temperature
    )

    population: List[RSASample] = []
    population_set = set()
    trace_steps = []
    population_predictor = dspy.ChainOfThought(RSAPopulation)
    aggregate_predictor = dspy.ChainOfThought(RSAggregate)
    population_instruction = (
        "Solve the problem step by step. Provide your reasoning and the final answer."
    )
    aggregate_instruction = (
        "You are given a problem and candidate solutions with reasoning. "
        "Some candidates may be incorrect or incomplete. "
        "Aggregate the useful ideas and intermediate steps, resolve disagreements, "
        "and produce a single high-quality solution with reasoning and a final answer. "
        "If all candidates are wrong, solve the problem from scratch."
    )

    max_attempts = max(config.population * 5, config.population + 5)
    attempts = 0
    parallel = max(1, config.parallel)
    executor = ThreadPoolExecutor(max_workers=parallel)
    try:
        while len(population) < config.population and attempts < max_attempts:
            remaining = max_attempts - attempts
            batch = min(parallel, remaining)
            futures = []
            for _ in range(batch):
                if config.debug:
                    print(f"[debug] population rollout_id={rollout_id}")
                rid = rollout_id
                rollout_id += 1
                attempts += 1
                futures.append(
                    executor.submit(
                        population_predictor,
                        instruction=population_instruction,
                        problem=prompt,
                        temperature=population_temperature,
                        max_tokens=config.max_tokens,
                        rollout_id=rid,
                    )
                )
            for future in as_completed(futures):
                result = future.result()
                sample = RSASample(
                    reasoning=(getattr(result, "reasoning", "") or "").strip(),
                    answer=(result.answer or "").strip(),
                )
                if sample.answer and sample.answer not in population_set:
                    population.append(sample)
                    population_set.add(sample.answer)
                    if config.completion_cb:
                        config.completion_cb(
                            {
                                "phase": "population",
                                "index": len(population),
                                "target": config.population,
                                "sample": sample,
                            }
                        )
                if len(population) >= config.population:
                    break

        if not population:
            raise RuntimeError(
                "Population is empty. Try higher temperature or a new prompt."
            )

        if config.progress_cb:
            config.progress_cb(
                {
                    "type": "init",
                    "population": population,
                    "cost": total_cost_usd(lm),
                }
            )

        trace_steps.append(
            {
                "step": 0,
                "population": [sample.__dict__ for sample in population],
                "cost": total_cost_usd(lm),
            }
        )

        for step in range(1, config.steps + 1):
            if len(population) < config.subset:
                if config.debug:
                    print(
                        "Stopping early: population smaller than subset "
                        f"({len(population)} < {config.subset})."
                    )
                if config.progress_cb:
                    config.progress_cb(
                        {
                            "type": "early_stop",
                            "step": step,
                            "population": population,
                            "cost": total_cost_usd(lm),
                        }
                    )
                break
            new_population: List[RSASample] = []
            new_population_set = set()
            attempts = 0
            while len(new_population) < config.population and attempts < max_attempts:
                remaining = max_attempts - attempts
                batch = min(parallel, remaining)
                futures = []
                for _ in range(batch):
                    subset = random.sample(population, config.subset)
                    if config.debug:
                        print(f"[debug] aggregate rollout_id={rollout_id}")
                    rid = rollout_id
                    rollout_id += 1
                    attempts += 1
                    futures.append(
                        executor.submit(
                            _call_predict,
                            aggregate_predictor,
                            instruction=aggregate_instruction,
                            problem=prompt,
                            candidates=subset,
                            temperature=aggregate_temperature,
                            max_tokens=config.max_tokens,
                            rollout_id=rid,
                        )
                    )
                for future in as_completed(futures):
                    candidate = future.result()
                    if candidate.answer and candidate.answer not in new_population_set:
                        new_population.append(candidate)
                        new_population_set.add(candidate.answer)
                        if config.completion_cb:
                            config.completion_cb(
                                {
                                    "phase": "aggregate",
                                    "step": step,
                                    "index": len(new_population),
                                    "target": config.population,
                                    "sample": candidate,
                                }
                            )
                    if len(new_population) >= config.population:
                        break
            population = new_population
            population_set = new_population_set
            if config.progress_cb:
                config.progress_cb(
                    {
                        "type": "step",
                        "step": step,
                        "population": population,
                        "cost": total_cost_usd(lm),
                    }
                )
            trace_steps.append(
                {
                    "step": step,
                    "population": [sample.__dict__ for sample in population],
                    "cost": total_cost_usd(lm),
                }
            )
    finally:
        executor.shutdown(wait=True)

    final_sample = random.choice(population)
    total_cost = total_cost_usd(lm)

    return RSAResult(
        answer=final_sample.answer.strip(),
        population=population,
        trace=trace_steps,
        total_cost=total_cost,
    )


def write_trace_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
