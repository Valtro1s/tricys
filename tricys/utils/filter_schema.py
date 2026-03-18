from typing import Any, Dict, List

import pandas as pd


def find_filter_schema_violations(
    df: pd.DataFrame, filter_schema: List[Dict[str, Any]] | None
) -> List[Dict[str, Any]]:
    """Evaluate filter_schema rules against a single-job result DataFrame.

    Each rule follows the same structure used by static alarm checks:
    {"columns": ["var1", ...], "min": value, "max": value}
    """
    if df.empty or not filter_schema:
        return []

    violations = []

    for rule in filter_schema:
        min_val = rule.get("min")
        max_val = rule.get("max")
        columns_to_check = rule.get("columns", [])

        for column_name in columns_to_check:
            if column_name not in df.columns:
                continue

            column_series = df[column_name]
            if not pd.api.types.is_numeric_dtype(column_series):
                continue

            if max_val is not None:
                peak_value = column_series.max()
                if pd.notna(peak_value) and peak_value > max_val:
                    violations.append(
                        {
                            "column": column_name,
                            "kind": "max",
                            "threshold": max_val,
                            "observed": float(peak_value),
                        }
                    )

            if min_val is not None:
                dip_value = column_series.min()
                if pd.notna(dip_value) and dip_value < min_val:
                    violations.append(
                        {
                            "column": column_name,
                            "kind": "min",
                            "threshold": min_val,
                            "observed": float(dip_value),
                        }
                    )

    return violations
