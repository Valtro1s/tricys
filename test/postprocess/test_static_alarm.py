import json
import os
import shutil

import pandas as pd
import pytest

from tricys.postprocess.static_alarm import check_thresholds

# --- Fixtures and Setup ---
TEST_DIR = "temp_static_alarm_test"


def create_hdf5_fixture(file_path, jobs_df, results_df):
    with pd.HDFStore(file_path, mode="w") as store:
        store.put("jobs", jobs_df, format="table", data_columns=True)
        store.append("results", results_df, index=False, data_columns=True)


@pytest.fixture(autouse=True)
def setup_and_teardown_test_dir():
    """Fixture to create and cleanup the test directory for each test."""
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)
    yield
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)


# --- Tests for static_alarm.py ---


@pytest.mark.build_test
def test_check_thresholds_no_alarm():
    """Tests that no alarm is triggered when data is within thresholds."""
    h5_path = os.path.join(TEST_DIR, "results.h5")
    jobs_df = pd.DataFrame({"job_id": [1]})
    df = pd.DataFrame(
        {
            "time": [0, 1, 2, 3],
            "var1": [1, 2, 3, 4],
            "var2": [10, 20, 30, 40],
            "job_id": [1, 1, 1, 1],
        }
    )
    create_hdf5_fixture(h5_path, jobs_df, df)
    rules = [{"columns": ["var1"], "min": 0, "max": 5}]
    check_thresholds(h5_path, TEST_DIR, rules)

    report_path = os.path.join(TEST_DIR, "alarm_report.json")
    assert os.path.exists(report_path)
    with open(report_path, "r") as f:
        report = json.load(f)
    assert not any(item["has_alarm"] for item in report if item["variable"] == "var1")


@pytest.mark.build_test
def test_check_thresholds_max_exceeded():
    """Tests that an alarm is triggered when max threshold is exceeded."""
    h5_path = os.path.join(TEST_DIR, "results.h5")
    jobs_df = pd.DataFrame({"job_id": [1]})
    df = pd.DataFrame(
        {"time": [0, 1, 2, 3], "var1": [1, 6, 3, 4], "job_id": [1, 1, 1, 1]}
    )
    create_hdf5_fixture(h5_path, jobs_df, df)
    rules = [{"columns": ["var1"], "min": 0, "max": 5}]
    check_thresholds(h5_path, TEST_DIR, rules)

    report_path = os.path.join(TEST_DIR, "alarm_report.json")
    with open(report_path, "r") as f:
        report = json.load(f)
    assert any(item["has_alarm"] for item in report if item["variable"] == "var1")


@pytest.mark.build_test
def test_check_thresholds_min_exceeded():
    """Tests that an alarm is triggered when min threshold is exceeded."""
    h5_path = os.path.join(TEST_DIR, "results.h5")
    jobs_df = pd.DataFrame({"job_id": [1]})
    df = pd.DataFrame(
        {"time": [0, 1, 2, 3], "var1": [1, -1, 3, 4], "job_id": [1, 1, 1, 1]}
    )
    create_hdf5_fixture(h5_path, jobs_df, df)
    rules = [{"columns": ["var1"], "min": 0, "max": 5}]
    check_thresholds(h5_path, TEST_DIR, rules)

    report_path = os.path.join(TEST_DIR, "alarm_report.json")
    with open(report_path, "r") as f:
        report = json.load(f)
    assert any(item["has_alarm"] for item in report if item["variable"] == "var1")


@pytest.mark.build_test
def test_check_thresholds_sweep_task():
    """Tests threshold checking for parameter sweep results."""
    h5_path = os.path.join(TEST_DIR, "results.h5")
    jobs_df = pd.DataFrame({"job_id": [1, 2], "p": ["1", "2"]})
    df = pd.DataFrame(
        {
            "time": [0, 1, 2, 0, 1, 2],
            "var": [1, 2, 3, 4, 8, 6],
            "job_id": [1, 1, 1, 2, 2, 2],
        }
    )
    create_hdf5_fixture(h5_path, jobs_df, df)
    rules = [{"columns": ["var"], "min": 0, "max": 5}]
    check_thresholds(h5_path, TEST_DIR, rules)

    report_path = os.path.join(TEST_DIR, "alarm_report.json")
    with open(report_path, "r") as f:
        report = json.load(f)

    alarm_p1 = next(item["has_alarm"] for item in report if item["p"] == "1")
    alarm_p2 = next(item["has_alarm"] for item in report if item["p"] == "2")

    assert not alarm_p1
    assert alarm_p2
