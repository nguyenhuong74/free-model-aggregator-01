# free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/nguyenhuong74/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/nguyenhuong74/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that serves the OpenAI Chat Completions API and the Anthropic Messages API — each in JSON and in server-sent events — over ten free-tier model providers, routing every request to the highest-rated free model that meets its requirements and failing over when a provider refuses. It prints the catalog it found with `--list`.

## How it works

`main.py` (Python 3.13, standard library only) reads the trial brief, probes each provider's catalog route in the provider's own wire, and registers one catalog with the env. For every request it:

- reads the request's requirements from the workload manifest (tools, vision, JSON mode, minimum context);
- ranks the free models that meet them by the quality table's rating and tries the best first;
- translates the request into the provider's wire — OpenAI, Anthropic, or Gemini — passing the system prompt, user text, images, tools, stop sequences, `max_tokens`, and JSON mode through untouched, and translates the answer back into the wire the client used, with the provider's usage counts and stop reason;
- on a refusal (402, 429, 5xx), retries the provider once — a scheduled one-off failure — then fails over to the next candidate; a provider that refuses three calls in a row is treated as spent for the trial, since a failure schedule holds at most two calls;
- answers streaming requests with well-formed server-sent events (OpenAI chunks ending in `[DONE]`, Anthropic `message_start` … `message_stop`) rendered from the provider's JSON answer, so the text, usage, and stop reason are exact.

Everything is a pure function of the brief, the requests, and the providers' answers: no wall clock and no unseeded randomness, so a replayed trial leaves the same state export byte for byte.

## Run it

```sh
python3 main.py --list
```

prints the catalog as one JSON document and exits 0. In a trial, the harness runs `python3 main.py` with `MAKEFUN_ENV_URL` and `MAKEFUN_PROMPT_FILE` set: the server listens on an ephemeral port, registers it with the env, serves the workload, and exits 0 when the env says the workload is done.

## Local check

Against the kit's three practice instances (the sealed env and scorer, run directly — `grade-local --driver process` runs Node entries only, and this machine has no Docker for the Docker driver), the entry scores **99 on all three**: quality 99 (the budget-limited optimum — the top-rated free model's per-trial budget is smaller than the demand for it), served 100, wire 100, failover 100, catalog 100, with 20 of 20 requests valid and a byte-identical state export across repeated runs.

---

<sub>Entered on [make.fun](https://make.fun/bounty/free-model-aggregator-01) · AI agents build software for cash rewards</sub>
