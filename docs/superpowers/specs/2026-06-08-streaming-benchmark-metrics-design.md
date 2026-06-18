# Streaming Benchmark Metrics Enhancement Design

**Goal:** Enhance the benchmark harness so it measures chat-style streaming inference rather than treating streaming as an optional mode.

**Scope:** This design updates benchmark metrics, runner output, CLI defaults, tests, and reporting language. It does not add non-streaming benchmark support, Prometheus integration, or model parameter sweep automation.

## Background

The MVP benchmark currently records request latency, TTFT, output bytes, and error state. This is enough for a smoke test, but not enough to evaluate dialogue-serving behavior. In a real chat product, users care about when output starts, whether chunks arrive smoothly, and whether the stream completes reliably.

The benchmark should therefore treat `stream=True` as the primary and only benchmark mode for this phase.

## Design Decisions

### Streaming-only benchmark path

`benchmark/harness/runner.py` will remove the runtime branch that supports non-streaming execution. Every benchmark request will send:

```json
{
  "messages": [{"role": "user", "content": "..."}],
  "max_tokens": 512,
  "stream": true
}
```

`benchmark/harness/cli.py` will default to streaming-only behavior and no longer expose `--stream` as a meaningful switch.

### Per-request metrics

`RequestResult` will include:

- `id`
- `ok`
- `latency_seconds`
- `ttft_seconds`
- `chunk_count`
- `first_chunk_bytes`
- `output_bytes`
- `inter_chunk_gap_avg_seconds`
- `inter_chunk_gap_p95_seconds`
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `tokens_per_second`
- `tpot_seconds`
- `error`

Token fields remain `null` unless the stream contains real OpenAI-compatible `usage` data.

### Token extraction rule

The harness will parse SSE `data:` lines from streaming responses. It will only use token counts when a parsed JSON object contains:

```json
{
  "usage": {
    "prompt_tokens": 1,
    "completion_tokens": 2,
    "total_tokens": 3
  }
}
```

If usage is absent, malformed, or not emitted by the server, token fields stay `null`. The benchmark must not estimate tokens from bytes, characters, or words.

### Chunk timing

The runner will record the timestamp of each received chunk. It will derive:

- TTFT: first chunk timestamp minus request start.
- Total latency: request end minus request start.
- Inter-chunk gaps: differences between consecutive chunk timestamps.
- Average gap and P95 gap per request.

This gives a more realistic view of streaming smoothness than total latency alone.

### Summary output

`run_benchmark()` will continue returning:

```json
{
  "summary": {...},
  "results": [...]
}
```

The summary will include:

- request counts and error counts
- requests per second
- latency summary
- TTFT summary
- output bytes summary
- chunk count summary
- inter-chunk gap summary
- measured token request count
- completion token summary
- tokens per second summary
- TPOT summary

Only successful requests contribute to latency/chunk summaries. Only successful requests with real `completion_tokens` contribute to token summaries.

## Error Handling

The runner will treat each request independently:

- HTTP status errors produce `ok=false` and record the HTTP error string.
- Timeout/network errors produce `ok=false` and record the exception string.
- SSE parse errors for individual chunks will not fail the request; invalid chunks are ignored for token extraction but still count toward bytes/chunk timing.
- A stream that returns bytes but no token usage is still successful if the HTTP stream completes normally.

## Tests

Add or update tests for:

- extracting token usage from valid SSE `data:` JSON lines
- returning `None` token stats when usage is absent
- computing chunk count, first chunk bytes, average gap, and P95 gap
- ensuring benchmark summary separates token-measured requests from normal successful requests

Existing gateway and workload tests should continue passing.

## Documentation Updates

Update `docs/mvp-report.md` and learning notes to explain:

- benchmark is streaming-first
- TTFT and chunk rhythm are primary dialogue-serving indicators
- TPS/TPOT are reported only when server-provided usage exists
- bytes are not treated as tokens

## Non-goals

- Do not implement non-streaming benchmark metrics in this phase.
- Do not estimate tokens locally.
- Do not add tokenizer dependencies.
- Do not add Prometheus/Grafana dashboards.
- Do not automate parameter sweep matrices yet.

## Self-review

- Placeholder scan: no placeholder requirements remain.
- Internal consistency: all token metrics depend on real `usage` data; no section suggests estimating tokens.
- Scope check: focused on benchmark metrics only; 14B model setup and parameter sweeps remain separate work.
- Ambiguity check: streaming-only means CLI benchmark requests always use `stream=true` for this phase.
