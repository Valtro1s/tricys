"""Single-run simulation executor for Monte Carlo workflows."""

import json
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from tricys.analysis.metric import calculate_single_job_metrics
from tricys.simulation.simulation import run_simulation as tricys_run_simulation
from tricys.utils.config_utils import basic_prepare_config
from tricys.utils.log_utils import setup_logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_METRICS_DEFINITION = {
    "Startup_Inventory": {
        "source_column": "sds.I[1]",
        "method": "calculate_startup_inventory",
    }
}

DEFAULT_BASE_CONFIG = {
    "paths": {
        "package_path": "example/example_model_single/example_model.mo",
    },
    "simulation": {
        "model_name": "example_model.Cycle",
        "variableFilter": "time|sds.I[1]",
        "stop_time": 12000.0,
        "step_size": 0.5,
        "concurrent": False,
        "keep_temp_files": False,
    },
    "metrics_definition": DEFAULT_METRICS_DEFINITION,
}


class SimulationExecutor:
    """Execute one TRICYS simulation and return startup inventory in grams.

    By default, this executor uses the bundled example model so the module is
    runnable out of the box. For a real CFEDR workflow, pass a project-specific
    base configuration or a JSON config path when constructing the executor.
    """

    def __init__(
        self,
        base_config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        base_dir: Optional[str] = None,
    ) -> None:
        if base_config is not None and config_path is not None:
            raise ValueError("Provide either base_config or config_path, not both.")

        if config_path is not None:
            with open(config_path, "r", encoding="utf-8") as file:
                loaded_config = json.load(file)
            self.base_dir = str(Path(config_path).resolve().parent)
        else:
            loaded_config = deepcopy(base_config or DEFAULT_BASE_CONFIG)
            self.base_dir = base_dir or str(PROJECT_ROOT)

        self.base_config = loaded_config

    def run_simulation(self, params_dict: Dict[str, Any]) -> float:
        """Run a single simulation and return Startup_Inventory in grams."""
        self._validate_params_dict(params_dict)

        runtime_input = self._build_runtime_input(params_dict)
        prepared_config, original_config = basic_prepare_config(
            runtime_input, base_dir=self.base_dir
        )
        setup_logging(prepared_config, original_config)

        tricys_run_simulation(prepared_config, export_csv=False)

        startup_inventory = self._extract_startup_inventory(
            prepared_config["paths"]["results_dir"],
            prepared_config.get("metrics_definition", DEFAULT_METRICS_DEFINITION),
        )
        if startup_inventory is None:
            raise RuntimeError("Failed to extract Startup_Inventory from simulation results.")
        return startup_inventory

    def run_simulation_batch(self, params_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run a batch of simulations and return per-job startup inventory records."""
        self._validate_params_list(params_list)

        runtime_input, jobs_file_path = self._build_runtime_batch_input(params_list)
        try:
            prepared_config, original_config = basic_prepare_config(
                runtime_input, base_dir=self.base_dir
            )
            setup_logging(prepared_config, original_config)

            tricys_run_simulation(prepared_config, export_csv=False)

            return self._extract_batch_results(
                prepared_config["paths"]["results_dir"],
                prepared_config.get("metrics_definition", DEFAULT_METRICS_DEFINITION),
                params_list,
            )
        finally:
            jobs_file = Path(jobs_file_path)
            if jobs_file.exists():
                jobs_file.unlink()

    def _build_runtime_input(self, params_dict: Dict[str, Any]) -> Dict[str, Any]:
        config = deepcopy(self.base_config)
        config.setdefault("simulation", {})
        config["simulation"]["concurrent"] = False
        metrics_definition = deepcopy(config.get("metrics_definition", {}))
        for metric_name, default_definition in DEFAULT_METRICS_DEFINITION.items():
            existing_definition = deepcopy(metrics_definition.get(metric_name, {}))
            merged_definition = deepcopy(default_definition)
            merged_definition.update(existing_definition)
            metrics_definition[metric_name] = merged_definition
        config["metrics_definition"] = metrics_definition

        simulation_parameters = deepcopy(config.get("simulation_parameters", {}))
        simulation_parameters.update(params_dict)
        config["simulation_parameters"] = simulation_parameters
        return config

    def _build_runtime_batch_input(
        self, params_list: List[Dict[str, Any]]
    ) -> tuple[Dict[str, Any], str]:
        config = deepcopy(self.base_config)
        config.setdefault("simulation", {})
        config["simulation"]["concurrent"] = len(params_list) > 1
        metrics_definition = deepcopy(config.get("metrics_definition", {}))
        for metric_name, default_definition in DEFAULT_METRICS_DEFINITION.items():
            existing_definition = deepcopy(metrics_definition.get(metric_name, {}))
            merged_definition = deepcopy(default_definition)
            merged_definition.update(existing_definition)
            metrics_definition[metric_name] = merged_definition
        config["metrics_definition"] = metrics_definition

        base_parameters = deepcopy(config.get("simulation_parameters", {}))
        batch_rows = []
        for params_dict in params_list:
            row = deepcopy(base_parameters)
            row.update(params_dict)
            batch_rows.append(row)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8", newline=""
        ) as file:
            pd.DataFrame(batch_rows).to_csv(file.name, index=False)
            jobs_file_path = file.name

        config["simulation_parameters"] = {"file": jobs_file_path}
        return config, jobs_file_path

    @staticmethod
    def _validate_params_dict(params_dict: Dict[str, Any]) -> None:
        if not isinstance(params_dict, dict) or not params_dict:
            raise ValueError("params_dict must be a non-empty dictionary.")

        for key, value in params_dict.items():
            if isinstance(value, (list, tuple, set, dict)):
                raise ValueError(
                    f"Parameter '{key}' must be a scalar value for single-run execution."
                )

    @staticmethod
    def _validate_params_list(params_list: List[Dict[str, Any]]) -> None:
        if not isinstance(params_list, list) or not params_list:
            raise ValueError("params_list must be a non-empty list of parameter dictionaries.")

        for index, params_dict in enumerate(params_list, start=1):
            try:
                SimulationExecutor._validate_params_dict(params_dict)
            except ValueError as error:
                raise ValueError(f"Invalid parameter set at index {index}: {error}") from error

    @staticmethod
    def _extract_startup_inventory(
        results_dir: str,
        metrics_definition: Dict[str, Any],
    ) -> Optional[float]:
        results_path = Path(results_dir)
        hdf_candidates = sorted(results_path.glob("sweep_results*.h5"))
        if not hdf_candidates:
            return None

        hdf_path = hdf_candidates[-1]
        with pd.HDFStore(hdf_path, mode="r") as store:
            if "/summary" in store.keys():
                summary_df = store.select("summary")
                if (
                    not summary_df.empty
                    and "Startup_Inventory" in summary_df.columns
                    and summary_df["Startup_Inventory"].notna().any()
                ):
                    value = summary_df["Startup_Inventory"].dropna().iloc[-1]
                    return float(value)

            if "/results" not in store.keys():
                return None

            results_df = store.select("results")
            if results_df.empty or "job_id" not in results_df.columns:
                return None

            last_job_id = int(results_df["job_id"].max())
            job_df = results_df[results_df["job_id"] == last_job_id].copy()
            metric_values = calculate_single_job_metrics(job_df, metrics_definition)
            startup_inventory = metric_values.get("Startup_Inventory")
            if startup_inventory is None:
                return None
            return float(startup_inventory)

    @staticmethod
    def _extract_batch_results(
        results_dir: str,
        metrics_definition: Dict[str, Any],
        params_list: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        results_path = Path(results_dir)
        hdf_candidates = sorted(results_path.glob("sweep_results*.h5"))
        if not hdf_candidates:
            raise RuntimeError("Failed to find HDF5 results for batch simulation.")

        hdf_path = hdf_candidates[-1]
        startup_inventory_by_job_id: Dict[int, float] = {}

        with pd.HDFStore(hdf_path, mode="r") as store:
            if "/summary" in store.keys():
                summary_df = store.select("summary")
                if not summary_df.empty and "Startup_Inventory" in summary_df.columns:
                    valid_rows = summary_df[["job_id", "Startup_Inventory"]].dropna(
                        subset=["Startup_Inventory"]
                    )
                    startup_inventory_by_job_id.update(
                        {
                            int(row.job_id): float(row.Startup_Inventory)
                            for row in valid_rows.itertuples(index=False)
                        }
                    )

            if not startup_inventory_by_job_id and "/results" in store.keys():
                results_df = store.select("results")
                if not results_df.empty and "job_id" in results_df.columns:
                    for job_id in sorted(results_df["job_id"].dropna().unique()):
                        job_df = results_df[results_df["job_id"] == job_id].copy()
                        metric_values = calculate_single_job_metrics(job_df, metrics_definition)
                        startup_inventory = metric_values.get("Startup_Inventory")
                        if startup_inventory is not None:
                            startup_inventory_by_job_id[int(job_id)] = float(startup_inventory)

        batch_records = []
        for job_id, params_dict in enumerate(params_list, start=1):
            startup_inventory = startup_inventory_by_job_id.get(job_id)
            record = {
                "status": "success" if startup_inventory is not None else "failed",
                "startup_inventory_g": startup_inventory,
                **params_dict,
            }
            if startup_inventory is None:
                record["error"] = "Startup_Inventory missing from batch results."
            batch_records.append(record)

        return batch_records


_DEFAULT_EXECUTOR = SimulationExecutor()


def run_simulation(params_dict: Dict[str, Any]) -> float:
    """Run one TRICYS simulation using the default executor configuration."""
    return _DEFAULT_EXECUTOR.run_simulation(params_dict)


__all__ = [
    "DEFAULT_BASE_CONFIG",
    "DEFAULT_METRICS_DEFINITION",
    "SimulationExecutor",
    "run_simulation",
]