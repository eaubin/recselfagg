[“Recursive Self-Aggregation Unlocks Deep Thinking in Large Language Models.”](https://rsa-llm.github.io/) implemented with DSPy.

## Setup

Set your API key (required for OpenAI models):

```bash
export OPENAI_API_KEY="..."
```

## Usage

No-install (using uvx from GitHub):

```bash
uvx --from git+https://github.com/eaubin/recselfagg recselfagg \
  --population 3 \
  --subset 2
  --steps 3
  --show-completions \
  "Argue for the best fictional character."
```

## CLI Flags

- `--model`: DSPy model id (default: `openai/gpt-4o-mini`)
- `--population`: population size `N`
- `--subset`: subset size `K` sampled each aggregation
- `--steps`: number of RSA steps
- `--temperature`: sampling temperature
- `--population-temperature`: temperature for population sampling (defaults to `--temperature`)
- `--aggregate-temperature`: temperature for aggregation steps (defaults to `--temperature`)
- `--max-tokens`: per-call token limit
- `--show-completions`: print every completion at each step
- `--seed`: random seed for reproducibility
- `--rollout-base`: base rollout id (defaults to random)
- `--no-progress`: disable progress logging
- `--debug`: print rollout ids and debug info for each call
- `--trace-json`: write JSON trace of populations at each step to a file
- `--final-population-file`: write final population (JSON) to a file
- `--parallel`: number of concurrent model calls (default: 1)

### Output

A random member of the final population is printed. 
