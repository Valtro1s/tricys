import argparse
import json
import logging
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from tricys.analysis.parameter_sampler import ParameterSampler
from tricys.analysis.simulation_executor import SimulationExecutor

logger = logging.getLogger(__name__)


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


class ProgressHeartbeat:
    """Print periodic progress while a long-running iteration is still executing."""

    def __init__(
        self,
        iteration: int,
        sample_count: int,
        start_time: float,
        heartbeat_seconds: float,
    ) -> None:
        self.iteration = iteration
        self.sample_count = sample_count
        self.start_time = start_time
        self.heartbeat_seconds = heartbeat_seconds
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self.heartbeat_seconds <= 0:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.heartbeat_seconds + 1)

    def _run(self) -> None:
        while not self._stop_event.wait(self.heartbeat_seconds):
            elapsed = time.perf_counter() - self.start_time
            logger.info(
                "Monte Carlo iteration still running",
                extra={
                    "iteration": self.iteration,
                    "sample_count": self.sample_count,
                    "elapsed_human": _format_duration(elapsed),
                    "elapsed_seconds": round(elapsed, 3),
                },
            )


class MonteCarloOrchestrator:
    """Coordinates parameter sampling and single-run TRICYS execution."""

    def __init__(
        self,
        sampler: Optional[ParameterSampler] = None,
        executor: Optional[SimulationExecutor] = None,
        output_dir: str = "monte_carlo_results",
        heartbeat_seconds: float = 15.0,
    ) -> None:
        self.sampler = sampler or ParameterSampler()
        self.executor = executor or SimulationExecutor()
        self.output_dir = Path(output_dir)
        self.heartbeat_seconds = heartbeat_seconds

    def run(self, sample_count: int) -> Dict[str, Any]:
        """Run Monte Carlo sampling and persist results to disk."""
        if sample_count <= 0:
            raise ValueError("sample_count must be a positive integer.")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        run_started_at = datetime.now()
        start_time = time.perf_counter()
        successful_results: List[float] = []
        records: List[Dict[str, Any]] = []
        failed_runs = 0

        logger.info("Starting Monte Carlo simulation", extra={"sample_count": sample_count})

        for index in range(sample_count):
            iteration = index + 1
            iteration_start = time.perf_counter()
            params = self.sampler.generate_sample()
            logger.info(
                "Monte Carlo iteration started",
                extra={
                    "iteration": iteration,
                    "sample_count": sample_count,
                    "progress_percent": round(iteration / sample_count * 100, 1),
                    "completed_runs": iteration - 1,
                },
            )
            heartbeat = ProgressHeartbeat(
                iteration=iteration,
                sample_count=sample_count,
                start_time=iteration_start,
                heartbeat_seconds=self.heartbeat_seconds,
            )
            heartbeat.start()

            try:
                startup_inventory = self.executor.run_simulation(params)
                successful_results.append(float(startup_inventory))
                records.append(
                    {
                        "iteration": iteration,
                        "status": "success",
                        "startup_inventory_g": float(startup_inventory),
                        **params,
                    }
                )
                current_status = "success"
            except Exception as error:
                failed_runs += 1
                logger.warning(
                    "Monte Carlo iteration failed",
                    extra={
                        "iteration": iteration,
                        "error": str(error),
                    },
                )
                records.append(
                    {
                        "iteration": iteration,
                        "status": "failed",
                        "startup_inventory_g": np.nan,
                        "error": str(error),
                        **params,
                    }
                )
                current_status = "failed"
            finally:
                heartbeat.stop()

            elapsed_iteration = time.perf_counter() - iteration_start
            remaining_runs = sample_count - iteration
            average_runtime = (time.perf_counter() - start_time) / iteration
            estimated_remaining = average_runtime * remaining_runs
            logger.info(
                "Monte Carlo iteration completed",
                extra={
                    "iteration": iteration,
                    "sample_count": sample_count,
                    "status": current_status,
                    "elapsed_seconds": round(elapsed_iteration, 3),
                    "elapsed_human": _format_duration(elapsed_iteration),
                    "success_count": len(successful_results),
                    "failed_count": failed_runs,
                    "estimated_remaining_human": _format_duration(estimated_remaining),
                },
            )

        total_elapsed = time.perf_counter() - start_time
        artifact_paths = self._save_results(
            records=records,
            successful_results=successful_results,
            sample_count=sample_count,
            failed_runs=failed_runs,
            run_started_at=run_started_at,
            total_elapsed=total_elapsed,
        )

        return {
            "results": successful_results,
            "records": records,
            "successful_runs": len(successful_results),
            "failed_runs": failed_runs,
            "elapsed_seconds": total_elapsed,
            "artifacts": artifact_paths,
        }

    def _save_results(
        self,
        records: List[Dict[str, Any]],
        successful_results: List[float],
        sample_count: int,
        failed_runs: int,
        run_started_at: datetime,
        total_elapsed: float,
    ) -> Dict[str, str]:
        timestamp = run_started_at.strftime("%Y%m%d_%H%M%S")
        csv_path = self.output_dir / f"results_{timestamp}.csv"
        npy_path = self.output_dir / f"results_{timestamp}.npy"
        json_path = self.output_dir / f"summary_{timestamp}.json"

        pd.DataFrame(records).to_csv(csv_path, index=False, encoding="utf-8-sig")
        np.save(npy_path, np.asarray(successful_results, dtype=float))

        summary = {
            "run_started_at": run_started_at.isoformat(),
            "requested_runs": sample_count,
            "successful_runs": len(successful_results),
            "failed_runs": failed_runs,
            "elapsed_seconds": total_elapsed,
            "mean_startup_inventory_g": (
                float(np.mean(successful_results)) if successful_results else None
            ),
            "std_startup_inventory_g": (
                float(np.std(successful_results)) if successful_results else None
            ),
            "csv_path": str(csv_path.resolve()),
            "npy_path": str(npy_path.resolve()),
        }

        with json_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

        logger.info(
            "Monte Carlo results saved",
            extra={
                "csv_path": str(csv_path),
                "npy_path": str(npy_path),
                "json_path": str(json_path),
            },
        )

        return {
            "csv": str(csv_path.resolve()),
            "npy": str(npy_path.resolve()),
            "json": str(json_path.resolve()),
        }


def configure_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
        force=True,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monte Carlo driver for TRICYS startup inventory analysis."
    )
    parser.add_argument(
        "N",
        type=int,
        help="Total number of Monte Carlo simulations to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="monte_carlo_results",
        help="Directory used to store CSV, NPY, and JSON outputs.",
    )
    parser.add_argument(
        "--sampler-config",
        type=str,
        default=None,
        help="Optional JSON file defining uncertain parameter distributions.",
    )
    parser.add_argument(
        "--simulation-config",
        type=str,
        default=None,
        help="Optional JSON file defining the base TRICYS simulation configuration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level, for example INFO or DEBUG.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=15.0,
        help="Seconds between progress heartbeats while a single iteration is still running.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    configure_logging(args.log_level)

    sampler = ParameterSampler(config_path=args.sampler_config, seed=args.seed)
    executor = SimulationExecutor(config_path=args.simulation_config)
    orchestrator = MonteCarloOrchestrator(
        sampler=sampler,
        executor=executor,
        output_dir=args.output_dir,
        heartbeat_seconds=args.heartbeat_seconds,
    )

    result = orchestrator.run(args.N)
    logger.info(
        "Monte Carlo run completed",
        extra={
            "successful_runs": result["successful_runs"],
            "failed_runs": result["failed_runs"],
            "elapsed_seconds": round(result["elapsed_seconds"], 3),
        },
    )


if __name__ == "__main__":
    main()