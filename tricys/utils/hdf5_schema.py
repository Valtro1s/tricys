import pandas as pd

CONFIG_KEY = "config"
JOBS_KEY = "jobs"
LEGACY_JOBS_METADATA_KEY = "jobs_metadata"
LOG_KEY = "log"
RESULTS_KEY = "results"
SUMMARY_KEY = "summary"


def _has_store_key(store: pd.HDFStore, key: str) -> bool:
    return f"/{key}" in store.keys()


def get_jobs_key(store: pd.HDFStore) -> str:
    if _has_store_key(store, JOBS_KEY):
        return JOBS_KEY
    if _has_store_key(store, LEGACY_JOBS_METADATA_KEY):
        return LEGACY_JOBS_METADATA_KEY
    raise KeyError("Neither 'jobs' nor legacy 'jobs_metadata' exists in HDF5 store.")


def load_jobs_df(h5_path: str) -> pd.DataFrame:
    with pd.HDFStore(h5_path, mode="r") as store:
        jobs_key = get_jobs_key(store)
    return pd.read_hdf(h5_path, jobs_key)


def normalize_summary_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df

    long_format_cols = {"job_id", "metric_name", "metric_value"}
    if long_format_cols.issubset(summary_df.columns):
        summary_df = (
            summary_df.pivot_table(
                index="job_id",
                columns="metric_name",
                values="metric_value",
                aggfunc="first",
            )
            .reset_index()
            .rename_axis(columns=None)
        )

    if "job_id" in summary_df.columns:
        numeric_job_ids = pd.to_numeric(summary_df["job_id"], errors="coerce")
        if numeric_job_ids.notna().all():
            summary_df["job_id"] = numeric_job_ids.astype(int)
        summary_df = summary_df.sort_values("job_id").reset_index(drop=True)

    return summary_df


def load_summary_df(h5_path: str, where: str | None = None) -> pd.DataFrame:
    with pd.HDFStore(h5_path, mode="r") as store:
        if not _has_store_key(store, SUMMARY_KEY):
            return pd.DataFrame()

    if where:
        summary_df = pd.read_hdf(h5_path, SUMMARY_KEY, where=where)
    else:
        summary_df = pd.read_hdf(h5_path, SUMMARY_KEY)

    return normalize_summary_df(summary_df)
