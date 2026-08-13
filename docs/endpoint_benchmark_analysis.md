# Endpoint benchmark analysis

## Executive summary

The benchmark completed successfully, and the three Gemma endpoints were stable
at every tested concurrency level. However, the final capacity report is **not
certified** because throughput was still increasing substantially at the
configured ceiling of 32 concurrent requests per endpoint. The run therefore
demonstrates that concurrency 32 is safe under the benchmark workload, but it
does not establish the endpoints' maximum sustainable or optimal capacity.

The simultaneous three-endpoint test passed all of its gates. It sustained
5.250 requests per second with 96.95% of the throughput predicted from the
isolated tests, a 2.88x speedup over the best single endpoint, and no errors.

## Artifacts and test configuration

- Final report: [`data/gemma_capacity/endpoint_capacity.json`](../data/gemma_capacity/endpoint_capacity.json)
- Resumable checkpoint: [`data/gemma_capacity/endpoint_benchmark.jsonl`](../data/gemma_capacity/endpoint_benchmark.jsonl)
- Report creation time: 2026-08-12 10:20:12 UTC
- Measured duration per concurrency level: 300 seconds
- Warmup per concurrency level: 30 seconds
- Tested concurrency levels: 1, 2, 4, 8, 16, 24, and 32
- Production-shaped fixture bank: 80 requests

The checkpoint contains 21 isolated results and three combined results, all
with the same benchmark fingerprint as the final report.

## Isolated endpoint results

| Endpoint | Model | RPS at 32 | p50 latency | p95 latency | Gain from 24 to 32 | Selected concurrency | Converged |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| endpoint_1 | gemma-4-31B-v2 | 1.811 | 4.92 s | 44.80 s | 52.5% | 32 | No |
| endpoint_2 | gemma-4-31B | 1.822 | 4.86 s | 44.63 s | 30.2% | 32 | No |
| endpoint_3 | gemma-4-31B-v3 | 1.782 | 4.96 s | 44.93 s | 26.0% | 32 | No |

All isolated measurements had zero retryable errors, zero nonretryable errors,
zero HTTP 429 responses, and zero truncated completions. Every tested level was
classified as safe by the benchmark.

The final step in each throughput curve gained far more than the 5% convergence
limit. Consequently, the benchmark selected concurrency 32 because it was the
highest-throughput safe level, but marked each endpoint as nonconverged. This is
the sole reason the endpoint reports, and therefore the top-level report, were
not certified.

The endpoints performed similarly. Endpoint 2 had the highest isolated
throughput, but its advantage over the other endpoints was small and does not
support unequal routing weights based on this run alone.

## Combined endpoint result

| Metric | Result | Certification requirement | Outcome |
| --- | ---: | ---: | --- |
| Predicted throughput | 5.415 requests/s | N/A | N/A |
| Observed throughput | 5.250 requests/s | N/A | N/A |
| Efficiency ratio | 96.95% | At least 90% | Pass |
| Speedup over best isolated endpoint | 2.88x | At least 2.5x | Pass |
| Error rate | 0% | At most 0.5% | Pass |

The combined run used 32 concurrent requests on each endpoint, for 96 workers
in total. Per-endpoint results were:

| Endpoint | Requests | Throughput | p50 latency | p95 latency | Change from isolated |
| --- | ---: | ---: | ---: | ---: | ---: |
| endpoint_1 | 566 | 1.689 requests/s | 5.65 s | 45.47 s | -6.7% |
| endpoint_2 | 579 | 1.755 requests/s | 5.07 s | 44.80 s | -3.7% |
| endpoint_3 | 595 | 1.806 requests/s | 4.85 s | 44.15 s | +1.3% |

The aggregate loss relative to the isolated prediction was only 3.05%, which
indicates little cross-endpoint interference. Endpoint 3 was marginally fastest
in the combined test, reinforcing that the differences between endpoints are
small.

## Reliability and compatibility

Across the isolated runs, combined run, and oversized probes, the report
records 6,952 measured requests. All succeeded, with:

- Zero retryable and nonretryable errors
- Zero HTTP 429 responses
- Zero truncated completions
- Zero observed model mismatches
- Matching prompt-token counts across all three preflight checks
- All 12 oversized-input probes passing

The run processed approximately 16.69 million prompt tokens and 3.98 million
completion tokens. During the combined phase, the endpoints collectively
delivered about 12,517 prompt tokens per second and 2,943 completion tokens per
second. These figures describe endpoint calls and token processing, not
completed poems; the production pipeline may make multiple calls per poem.

## Latency interpretation

At concurrency 32, the overall p95 latency was approximately 45 seconds on each
endpoint. This is safe under the benchmark rule, which accepts p95 latency below
half of the configured 300-second request timeout, or 150 seconds. That rule is
generous and should not substitute for a product-specific latency objective.

Relative to concurrency 1, p95 latency at concurrency 32 increased by 65% on
endpoint 1, 82% on endpoint 2, and 65% on endpoint 3. The added throughput is
therefore accompanied by materially slower tail responses. If the production
system requires a lower p95, its operating concurrency may need to be below the
eventual throughput-optimal capacity.

At the selected level, generation and repair requests were the slowest request
types, with per-kind p95 values of roughly 34 to 47 seconds. Validation calls
were considerably faster, with p95 values of approximately 0.65 to 5.5 seconds.

## Measurement caveats

1. Concurrency levels 1 and 2 completed only 23 to 37 measured requests, fewer
   than the 80 fixtures in the workload bank. They therefore did not exercise a
   complete fixture cycle and should not be compared directly with the higher
   levels as if their request mix were identical.
2. Each level was measured once, so the report has no repeated-run variance or
   confidence intervals. Endpoint 1's uneven scaling around concurrency 24 is
   a reason to confirm the curve with another run.
3. Workers finish requests that are already in flight when the nominal
   five-minute deadline is reached. Reported durations are consequently longer
   than 300 seconds, although the throughput calculation uses the actual
   elapsed duration.
4. The benchmark's safety threshold is based on transport errors and a
   150-second p95 limit. A concurrency can pass that definition while still
   missing a stricter production latency target.

## Conclusions and recommendations

1. Treat 32 concurrent requests per endpoint as **tested safe**, not as
   certified or optimal capacity.
2. Increase each configured endpoint ceiling and test additional levels such as
   40, 48, and 64. Continue until the last safe step improves throughput by less
   than 5%, or until error rate or latency becomes unacceptable.
3. Repeat the high-concurrency measurements to check run-to-run stability before
   committing to a production limit.
4. Define an explicit production p95 latency objective. Use it alongside the
   benchmark's throughput and error gates when selecting the final concurrency.
5. Continue using all three endpoints together. The combined run passed its
   certification gates and achieved nearly linear aggregate scaling.

The current `certified: false` result should therefore be read as an incomplete
capacity search, not an endpoint reliability failure.
