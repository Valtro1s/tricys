import pandas as pd

from tricys.utils.filter_schema import find_filter_schema_violations


def test_find_filter_schema_violations_within_bounds():
    df = pd.DataFrame(
        {
            "time": [0, 1, 2],
            "var1": [1.0, 2.0, 3.0],
            "var2": [10.0, 11.0, 12.0],
        }
    )
    filter_schema = [{"columns": ["var1"], "min": 0.0, "max": 5.0}]

    violations = find_filter_schema_violations(df, filter_schema)

    assert violations == []


def test_find_filter_schema_violations_detects_upper_and_lower_bounds():
    df = pd.DataFrame(
        {
            "time": [0, 1, 2],
            "var1": [1.0, 6.0, 3.0],
            "var2": [10.0, -2.0, 12.0],
        }
    )
    filter_schema = [
        {"columns": ["var1"], "max": 5.0},
        {"columns": ["var2"], "min": 0.0},
    ]

    violations = find_filter_schema_violations(df, filter_schema)

    assert violations == [
        {"column": "var1", "kind": "max", "threshold": 5.0, "observed": 6.0},
        {"column": "var2", "kind": "min", "threshold": 0.0, "observed": -2.0},
    ]


def test_find_filter_schema_violations_ignores_missing_columns():
    df = pd.DataFrame({"time": [0, 1], "var1": [1.0, 2.0]})
    filter_schema = [{"columns": ["missing_col"], "min": 0.0, "max": 1.0}]

    violations = find_filter_schema_violations(df, filter_schema)

    assert violations == []
