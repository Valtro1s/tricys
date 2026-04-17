import numpy as np
import pandas as pd

from analyze_results import analyze_results_file, compute_statistics, load_results


def test_load_results_from_npy(tmp_path):
    input_path = tmp_path / "results.npy"
    np.save(input_path, np.array([10.0, 20.0, 30.0]))

    results = load_results(str(input_path))

    assert np.array_equal(results, np.array([10.0, 20.0, 30.0]))


def test_load_results_from_csv_with_status_column(tmp_path):
    input_path = tmp_path / "results.csv"
    pd.DataFrame(
        {
            "iteration": [1, 2, 3],
            "status": ["success", "failed", "success"],
            "startup_inventory_g": [100.0, np.nan, 140.0],
        }
    ).to_csv(input_path, index=False)

    results = load_results(str(input_path))

    assert np.array_equal(results, np.array([100.0, 140.0]))


def test_compute_statistics():
    stats = compute_statistics(np.array([100.0, 120.0, 140.0, 160.0]))

    assert stats["count"] == 4
    assert stats["mean"] == 130.0
    assert stats["min"] == 100.0
    assert stats["max"] == 160.0


def test_analyze_results_file_generates_artifacts(tmp_path):
    input_path = tmp_path / "results.npy"
    output_dir = tmp_path / "analysis"
    np.save(input_path, np.array([100.0, 120.0, 140.0, 160.0]))

    artifacts = analyze_results_file(str(input_path), str(output_dir))

    report_path = output_dir / "report.txt"
    figure_path = output_dir / "uncertainty_analysis.png"

    assert artifacts["report"] == str(report_path.resolve())
    assert artifacts["figure"] == str(figure_path.resolve())
    assert report_path.exists()
    assert figure_path.exists()
    assert "95%置信区间" in report_path.read_text(encoding="utf-8")