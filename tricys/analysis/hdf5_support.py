from typing import Dict, Iterable, Iterator, Optional

import pandas as pd

from tricys.utils.hdf5_schema import RESULTS_KEY, load_jobs_df


def build_series_label(variable_name: str, job_params: Dict[str, object]) -> str:
    filtered_params = {k: v for k, v in sorted(job_params.items()) if pd.notna(v)}
    param_string = "&".join([f"{k}={v}" for k, v in filtered_params.items()])
    return f"{variable_name}&{param_string}" if param_string else variable_name


def get_hdf5_result_columns(h5_path: str) -> list[str]:
    with pd.HDFStore(h5_path, mode="r") as store:
        sample_df = store.select(RESULTS_KEY, start=0, stop=1)
    if sample_df.empty:
        return []
    return [col for col in sample_df.columns if col not in ["time", "job_id"]]


def iter_hdf5_job_results(
    h5_path: str,
    jobs_df: Optional[pd.DataFrame] = None,
    selected_job_ids: Optional[Iterable[int]] = None,
    columns: Optional[list[str]] = None,
) -> Iterator[tuple[int, Dict[str, object], pd.DataFrame]]:
    if jobs_df is None:
        jobs_df = load_jobs_df(h5_path)

    if selected_job_ids is not None:
        selected_job_ids = {int(job_id) for job_id in selected_job_ids}
        jobs_df = jobs_df[jobs_df["job_id"].isin(selected_job_ids)]

    read_columns = None
    if columns:
        read_columns = list(dict.fromkeys(["time", "job_id"] + columns))

    with pd.HDFStore(h5_path, mode="r") as store:
        for _, job_row in jobs_df.iterrows():
            job_id = int(job_row["job_id"])
            job_params = job_row.drop(labels=["job_id"]).to_dict()
            job_df = store.select(
                RESULTS_KEY,
                where=f"job_id == {job_id}",
                columns=read_columns,
            )
            if job_df.empty:
                continue
            yield job_id, job_params, job_df.reset_index(drop=True)


def _build_sample_dataframe(
    time_values: list[float], sample_columns: Dict[str, list[float]]
) -> pd.DataFrame:
    sample_df = pd.DataFrame({"time": pd.Series(time_values)})
    for label, values in sample_columns.items():
        sample_df[label] = pd.Series(values)
    return sample_df


def build_dynamic_slices_from_hdf5(
    h5_path: str,
    jobs_df: Optional[pd.DataFrame] = None,
    reference_variable: Optional[str] = None,
    num_points: int = 20,
    interval: int = 2,
) -> dict:
    if jobs_df is None:
        jobs_df = load_jobs_df(h5_path)

    result_columns = get_hdf5_result_columns(h5_path)
    if not result_columns or jobs_df.empty:
        return {}

    if reference_variable not in result_columns:
        reference_variable = result_columns[len(result_columns) // 2]

    window_size = (num_points - 1) * interval + 1
    start_sample_columns = {}
    end_sample_columns = {}
    turning_sample_columns = {}
    start_time_values = []
    end_time_values = []
    turning_time_values = []
    reference_label = None
    turning_index = None
    best_turning_value = float("inf")

    for _, job_params, job_df in iter_hdf5_job_results(
        h5_path, jobs_df=jobs_df, columns=result_columns
    ):
        time_series = job_df["time"].reset_index(drop=True)
        start_indices = list(range(0, min(len(job_df), window_size), interval))
        end_start = max(0, len(job_df) - window_size)
        end_indices = list(range(end_start, len(job_df), interval))

        if not start_time_values:
            start_time_values = time_series.iloc[start_indices].tolist()
        if not end_time_values:
            end_time_values = time_series.iloc[end_indices].tolist()

        for variable_name in result_columns:
            if variable_name not in job_df.columns:
                continue

            label = build_series_label(variable_name, job_params)
            series = job_df[variable_name].reset_index(drop=True)
            start_sample_columns[label] = series.iloc[start_indices].tolist()
            end_sample_columns[label] = series.iloc[end_indices].tolist()

            if variable_name == reference_variable:
                current_min = float(series.min())
                if current_min < best_turning_value:
                    best_turning_value = current_min
                    turning_index = int(series.idxmin())
                    reference_label = label

    if turning_index is not None:
        window_radius_indices = (num_points // 2) * interval
        for _, job_params, job_df in iter_hdf5_job_results(
            h5_path, jobs_df=jobs_df, columns=result_columns
        ):
            time_series = job_df["time"].reset_index(drop=True)
            start_idx = max(0, turning_index - window_radius_indices)
            end_idx = min(len(job_df), turning_index + window_radius_indices)
            turning_indices = list(range(start_idx, end_idx, interval))

            if not turning_time_values:
                turning_time_values = time_series.iloc[turning_indices].tolist()

            for variable_name in result_columns:
                if variable_name not in job_df.columns:
                    continue
                label = build_series_label(variable_name, job_params)
                series = job_df[variable_name].reset_index(drop=True)
                turning_sample_columns[label] = series.iloc[turning_indices].tolist()

    return {
        "reference_label": reference_label,
        "reference_variable": reference_variable,
        "start_sample_df": _build_sample_dataframe(
            start_time_values, start_sample_columns
        ),
        "end_sample_df": _build_sample_dataframe(end_time_values, end_sample_columns),
        "turning_sample_df": _build_sample_dataframe(
            turning_time_values, turning_sample_columns
        ),
    }
