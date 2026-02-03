from __future__ import annotations

import argparse
import sys

from .rsa import RSAConfig, RSAResult, run_rsa, write_trace_json


def _read_prompt(arg_prompt: str | None) -> str:
    if arg_prompt:
        return arg_prompt.strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    raise SystemExit("Prompt required. Pass it as an argument or via stdin.")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursive Self-Aggregation (RSA) CLI over a single prompt."
    )
    parser.add_argument("prompt", nargs="?", help="Prompt to solve.")
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="Model id.")
    parser.add_argument("--population", type=int, default=8, help="Population size N.")
    parser.add_argument("--subset", type=int, default=3, help="Subset size K.")
    parser.add_argument("--steps", type=int, default=2, help="Number of RSA steps.")
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature."
    )
    parser.add_argument(
        "--population-temperature",
        type=float,
        default=None,
        help="Temperature for population sampling (defaults to --temperature).",
    )
    parser.add_argument(
        "--aggregate-temperature",
        type=float,
        default=None,
        help="Temperature for aggregation steps (defaults to --temperature).",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None, help="Max tokens per LM call."
    )
    parser.add_argument(
        "--show-completions",
        action="store_true",
        help="Print all completions for the given parameters (debug only).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--rollout-base",
        type=int,
        default=None,
        help="Base rollout id (defaults to a random value).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress logs.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print rollout ids and debug info for each call.",
    )
    parser.add_argument(
        "--trace-json",
        type=str,
        default=None,
        help="Write JSON trace of populations at each step to this file.",
    )
    parser.add_argument(
        "--final-population-file",
        type=str,
        default=None,
        help="Write final population (JSON) to this file.",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Enable DSPy LM cache (disabled by default).",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of concurrent model calls (default: 1).",
    )
    return parser.parse_args(argv)


def _print_progress(cost: float, step: int, total_steps: int) -> None:
    print(
        f"Step {step}/{total_steps} complete. Total cost: ${cost:.6f}",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    prompt = _read_prompt(args.prompt)
    if not prompt:
        raise SystemExit("Prompt is empty.")

    def progress_cb(event: dict) -> None:
        if args.no_progress:
            return
        event_type = event.get("type")
        if event_type == "init":
            print(
                f"Initialized population: {len(event['population'])}",
                file=sys.stderr,
            )
        elif event_type == "step":
            _print_progress(event["cost"], event["step"], args.steps)
        elif event_type == "early_stop":
            print(
                "Stopping early: population smaller than subset "
                f"({len(event['population'])} < {args.subset}).",
                file=sys.stderr,
            )

    def completion_cb(event: dict) -> None:
        if not (args.show_completions and args.debug):
            return
        phase = event.get("phase")
        if phase == "population":
            header = "Initial Completion"
            idx = event["index"]
            total = event["target"]
        else:
            header = f"Step {event['step']} Completion"
            idx = event["index"]
            total = event["target"]
        sample = event["sample"]
        print(
            f"\n=== {header} {idx}/{total} ===",
            file=sys.stderr,
        )
        print("Reasoning:", file=sys.stderr)
        print(sample.reasoning.strip(), file=sys.stderr)
        print("\nAnswer:", file=sys.stderr)
        print(sample.answer.strip(), file=sys.stderr)

    config = RSAConfig(
        prompt=prompt,
        model=args.model,
        population=args.population,
        subset=args.subset,
        steps=args.steps,
        temperature=args.temperature,
        population_temperature=args.population_temperature,
        aggregate_temperature=args.aggregate_temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        rollout_base=args.rollout_base,
        debug=args.debug,
        cache=args.cache,
        parallel=args.parallel,
        progress_cb=progress_cb,
        completion_cb=completion_cb,
    )

    result = run_rsa(config)

    if args.debug:
        print("\n=== Final Answer ===", file=sys.stderr)
    print(result.answer)
    if args.debug:
        print("\n=== Cost Summary ===", file=sys.stderr)
        print(f"Total cost (USD): ${result.total_cost:.6f}", file=sys.stderr)

    if args.trace_json:
        write_trace_json(
            args.trace_json,
            {
                "model": args.model,
                "population": args.population,
                "subset": args.subset,
                "steps": args.steps,
                "trace": result.trace,
            },
        )

    if args.final_population_file:
        write_trace_json(
            args.final_population_file,
            [sample.__dict__ for sample in result.population],
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
