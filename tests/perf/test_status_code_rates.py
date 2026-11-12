"""Status-code-derived failure rates (follow-up to #2053: status-code-derived
error/429/5xx rates only — transport classes are out of scope, as the load path
does not record a failure kind).

Invariant under test: ``error_rate >= 429_rate + 5xx_rate``, with equality iff
every failure carried a status code. Rates are shares of total requests; a
run with failures but no classified ones reports ``-1`` (not applicable),
never ``0.0`` — 0.0 would read as "no throttling happened".
"""
import pytest

from evalscope.perf.utils.benchmark_util import BenchmarkData, MetricsAccumulator
from evalscope.perf.utils.perf_constants import Metrics


def _failed(status_code=None):
    data = BenchmarkData(request={})
    data.success = False
    data.start_time = 0.0
    data.completed_time = 1.0
    data.status_code = status_code
    data.error = 'failure'
    return data


def _ok():
    data = BenchmarkData(request={})
    data.success = True
    data.start_time = 0.0
    data.completed_time = 1.0
    data.chunk_times = [0.5]
    data.prompt_tokens = 10
    data.completion_tokens = 10
    return data


def _accumulate(*requests):
    acc = MetricsAccumulator()
    plugin = type('P', (), {'parse_responses': staticmethod(lambda *a, **k: (0, 0))})()
    for req in requests:
        acc.update(req, plugin)
    return acc


def test_rates_classified_by_status():
    result = _accumulate(
        _ok(),
        _failed(status_code=429),
        _failed(status_code=429),
        _failed(status_code=500),
        _failed(status_code=503),
    ).to_result()

    assert result.total_requests == 5
    assert result.rate_limited_requests == 2
    assert result.server_error_requests == 2
    assert result.classified_failures == 4
    assert result.rate_limited_rate == pytest.approx(40.0)  # 2 / 5
    assert result.server_error_rate == pytest.approx(40.0)  # 2 / 5


def test_unclassified_failures_show_not_applicable():
    # all failures are transport-level (no status code): the breakdown must show
    # -1 (not applicable), never 0.0 — 0.0 would read as "no throttling happened"
    result = _accumulate(_ok(), _failed(), _failed()).to_result()

    assert result.rate_limited_rate == -1
    assert result.server_error_rate == -1
    assert result.classified_failures == 0


def test_all_success_reports_zero_rates():
    result = _accumulate(_ok(), _ok()).to_result()
    assert result.rate_limited_rate == 0.0 and result.server_error_rate == 0.0


def test_invariant_error_rate_ge_sum_of_breakdown():
    # error_rate (failed/total) >= 429_rate + 5xx_rate, equality iff every failure
    # carried a status code — asserted on a mixed stream
    import random

    rng = random.Random(7)
    requests = []
    for i in range(60):
        if i % 5 == 0:
            requests.append(_failed(status_code=429 if i % 10 == 0 else 500))
        elif i % 7 == 0:
            requests.append(_failed())  # unclassified
        else:
            requests.append(_ok())
    result = _accumulate(*requests).to_result()

    failed = result.failed_requests
    error_rate = failed * 100.0 / result.total_requests
    assert result.rate_limited_rate + result.server_error_rate <= error_rate + 1e-9
    classified = result.rate_limited_requests + result.server_error_requests
    equality_holds = classified == failed
    assert (abs(result.rate_limited_rate + result.server_error_rate - error_rate) < 1e-9) == equality_holds


def test_empty_run_reports_not_applicable():
    result = _accumulate().to_result()
    assert result.rate_limited_rate == -1 and result.server_error_rate == -1
