import json
import time

import numpy as np
import pandas as pd

from monte_carlo_driver import MonteCarloOrchestrator, ProgressHeartbeat, _format_duration


class DummySampler:
    def __init__(self, samples):
        self.samples = list(samples)
        self.index = 0

    def generate_sample(self):
        sample = self.samples[self.index]
        self.index += 1
        return sample


class DummyExecutor:
    def __init__(self, responses):
        self.responses = list(responses)
        self.index = 0

    def run_simulation(self, params_dict):
        response = self.responses[self.index]
        self.index += 1
        if isinstance(response, Exception):
            raise response
        return response


class SlowDummyExecutor:
    def __init__(self, sleep_seconds, response):
        self.sleep_seconds = sleep_seconds
        self.response = response

    def run_simulation(self, params_dict):
        time.sleep(self.sleep_seconds)
        return self.response


def test_monte_carlo_orchestrator_saves_results(tmp_path):
    sampler = DummySampler(
        [
            {"blanket.TBR": 1.05},
            {"blanket.TBR": 1.10},
            {"blanket.TBR": 1.15},
        ]
    )
    executor = DummyExecutor([100.0, 120.0, 140.0])
    orchestrator = MonteCarloOrchestrator(
        sampler=sampler,
        executor=executor,
        output_dir=str(tmp_path),
    )

    result = orchestrator.run(3)

    assert result["successful_runs"] == 3
    assert result["failed_runs"] == 0
    assert result["results"] == [100.0, 120.0, 140.0]

    csv_path = result["artifacts"]["csv"]
    npy_path = result["artifacts"]["npy"]
    json_path = result["artifacts"]["json"]

    df = pd.read_csv(csv_path)
    assert len(df) == 3
    assert set(df["status"]) == {"success"}

    array = np.load(npy_path)
    assert np.array_equal(array, np.array([100.0, 120.0, 140.0]))

    with open(json_path, "r", encoding="utf-8") as file:
        summary = json.load(file)
    assert summary["requested_runs"] == 3
    assert summary["successful_runs"] == 3
    assert summary["failed_runs"] == 0


def test_monte_carlo_orchestrator_skips_failed_runs(tmp_path):
    sampler = DummySampler(
        [
            {"blanket.TBR": 1.05},
            {"blanket.TBR": 1.10},
            {"blanket.TBR": 1.15},
        ]
    )
    executor = DummyExecutor([100.0, RuntimeError("boom"), 140.0])
    orchestrator = MonteCarloOrchestrator(
        sampler=sampler,
        executor=executor,
        output_dir=str(tmp_path),
    )

    result = orchestrator.run(3)

    assert result["successful_runs"] == 2
    assert result["failed_runs"] == 1
    assert result["results"] == [100.0, 140.0]

    df = pd.read_csv(result["artifacts"]["csv"])
    assert len(df) == 3
    assert list(df["status"]) == ["success", "failed", "success"]
    assert df["startup_inventory_g"].isna().sum() == 1


def test_format_duration():
    assert _format_duration(8.4) == "8s"
    assert _format_duration(65.0) == "1m 05s"
    assert _format_duration(3665.0) == "1h 01m 05s"


def test_progress_heartbeat_emits_log(monkeypatch):
    captured = []

    def fake_info(message, extra=None):
        captured.append((message, extra))

    monkeypatch.setattr("monte_carlo_driver.logger.info", fake_info)

    heartbeat = ProgressHeartbeat(
        iteration=2,
        sample_count=10,
        start_time=time.perf_counter(),
        heartbeat_seconds=0.01,
    )
    heartbeat.start()
    time.sleep(0.03)
    heartbeat.stop()

    assert any(message == "Monte Carlo iteration still running" for message, _ in captured)


def test_monte_carlo_orchestrator_logs_iteration_started(monkeypatch, tmp_path):
    captured = []

    def fake_info(message, extra=None):
        captured.append((message, extra))

    monkeypatch.setattr("monte_carlo_driver.logger.info", fake_info)

    orchestrator = MonteCarloOrchestrator(
        sampler=DummySampler([{"blanket.TBR": 1.05}]),
        executor=SlowDummyExecutor(0.02, 100.0),
        output_dir=str(tmp_path),
        heartbeat_seconds=0.01,
    )

    orchestrator.run(1)

    assert any(message == "Monte Carlo iteration started" for message, _ in captured)
    assert any(message == "Monte Carlo iteration completed" for message, _ in captured)