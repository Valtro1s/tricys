import os
import shutil

import pandas as pd
import pytest

from tricys.postprocess.baseline_analysis import baseline_analysis

TEST_DIR = "temp_baseline_analysis_test"


def create_hdf5_fixture(file_path, jobs_df, results_df):
    with pd.HDFStore(file_path, mode="w") as store:
        store.put("jobs", jobs_df, format="table", data_columns=True)
        store.append("results", results_df, index=False, data_columns=True)


@pytest.fixture(autouse=True)
def setup_and_teardown_test_dir():
    """Set up and tear down the test directory."""
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)
    yield
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)


def test_baseline_analysis():
    """Test the baseline_analysis function."""
    h5_path = os.path.join(TEST_DIR, "results.h5")
    jobs_df = pd.DataFrame({"job_id": [1]})
    results_df = pd.DataFrame(
        {
            "time": range(10),
            "sds.I[1]": range(10, 20),
            "sds.I[2]": range(20, 30),
            "job_id": [1] * 10,
        }
    )
    create_hdf5_fixture(h5_path, jobs_df, results_df)

    output_dir = os.path.join(TEST_DIR, "output")

    baseline_analysis(h5_path, output_dir)

    report_dir = os.path.join(TEST_DIR, "report")
    assert os.path.exists(report_dir)

    plot1_path = os.path.join(report_dir, "simulation_all_curves_detailed.svg")
    plot2_path = os.path.join(report_dir, "final_values_bar_chart.svg")
    report_path = os.path.join(report_dir, "baseline_condition_analysis_report.md")

    assert os.path.exists(plot1_path)
    assert os.path.exists(plot2_path)
    assert os.path.exists(report_path)
