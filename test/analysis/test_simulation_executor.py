import pandas as pd
import pytest

from tricys.analysis.simulation_executor import SimulationExecutor


def test_extract_startup_inventory_from_summary(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    hdf_path = results_dir / "sweep_results.h5"

    with pd.HDFStore(hdf_path, mode="w") as store:
        store.put(
            "summary",
            pd.DataFrame(
                [
                    {"job_id": 1, "Startup_Inventory": 123.4},
                ]
            ),
        )

    value = SimulationExecutor._extract_startup_inventory(
        str(results_dir),
        {
            "Startup_Inventory": {
                "source_column": "sds.I[1]",
                "method": "calculate_startup_inventory",
            }
        },
    )

    assert value == 123.4


def test_extract_startup_inventory_falls_back_to_results(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    hdf_path = results_dir / "sweep_results.h5"

    with pd.HDFStore(hdf_path, mode="w") as store:
        store.put(
            "results",
            pd.DataFrame(
                {
                    "time": [0.0, 1.0, 2.0],
                    "sds.I[1]": [100.0, 80.0, 90.0],
                    "job_id": [1, 1, 1],
                }
            ),
        )

    value = SimulationExecutor._extract_startup_inventory(
        str(results_dir),
        {
            "Startup_Inventory": {
                "source_column": "sds.I[1]",
                "method": "calculate_startup_inventory",
            }
        },
    )

    assert value == 20.0


def test_run_simulation_builds_config_and_returns_metric(tmp_path, monkeypatch):
    base_config = {
        "paths": {"package_path": "example/example_model_single/example_model.mo"},
        "simulation": {
            "model_name": "example_model.Cycle",
            "variableFilter": "time|sds.I[1]",
            "stop_time": 10.0,
            "step_size": 1.0,
        },
    }
    executor = SimulationExecutor(base_config=base_config, base_dir=str(tmp_path))

    def fake_prepare_config(config_or_path, base_dir=None):
        prepared = config_or_path.copy()
        prepared["paths"] = prepared.get("paths", {}).copy()
        prepared["paths"]["results_dir"] = str(tmp_path / "run" / "results")
        prepared["paths"]["temp_dir"] = str(tmp_path / "run" / "temp")
        prepared["paths"]["log_dir"] = str(tmp_path / "run" / "log")
        return prepared, config_or_path

    def fake_setup_logging(config, original_config=None):
        return None

    def fake_tricys_run_simulation(config, export_csv=False):
        results_dir = tmp_path / "run" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        with pd.HDFStore(results_dir / "sweep_results.h5", mode="w") as store:
            store.put(
                "summary",
                pd.DataFrame(
                    [
                        {"job_id": 1, "Startup_Inventory": 321.0},
                    ]
                ),
            )
        assert config["simulation_parameters"]["blanket.TBR"] == 1.12
        assert config["simulation"]["concurrent"] is False

    monkeypatch.setattr(
        "tricys.analysis.simulation_executor.basic_prepare_config",
        fake_prepare_config,
    )
    monkeypatch.setattr(
        "tricys.analysis.simulation_executor.setup_logging",
        fake_setup_logging,
    )
    monkeypatch.setattr(
        "tricys.analysis.simulation_executor.tricys_run_simulation",
        fake_tricys_run_simulation,
    )

    value = executor.run_simulation({"blanket.TBR": 1.12})

    assert value == 321.0


def test_run_simulation_rejects_sweep_values():
    executor = SimulationExecutor()

    with pytest.raises(ValueError):
        executor.run_simulation({"blanket.TBR": [1.05, 1.10]})


def test_build_runtime_input_preserves_user_metric_source_column():
    executor = SimulationExecutor(
        base_config={
            "metrics_definition": {
                "Startup_Inventory": {
                    "source_column": "sds.inventory",
                }
            }
        }
    )

    runtime_input = executor._build_runtime_input({"bz.TBR": 1.1})

    assert runtime_input["metrics_definition"]["Startup_Inventory"]["source_column"] == "sds.inventory"
    assert runtime_input["metrics_definition"]["Startup_Inventory"]["method"] == "calculate_startup_inventory"