import argparse
import logging
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def configure_matplotlib_for_chinese() -> None:
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "DejaVu Sans",
        "Arial Unicode MS",
        "sans-serif",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def load_results(input_path: str) -> np.ndarray:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {input_path}")

    if path.suffix.lower() == ".npy":
        data = np.load(path)
        return _clean_result_array(data)

    if path.suffix.lower() == ".csv":
        data_frame = pd.read_csv(path)
        return _extract_results_from_csv(data_frame)

    raise ValueError("Only .npy and .csv result files are supported.")


def _clean_result_array(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data, dtype=float).reshape(-1)
    array = array[~np.isnan(array)]
    if array.size == 0:
        raise ValueError("No valid simulation results were found in the input file.")
    return array


def _extract_results_from_csv(data_frame: pd.DataFrame) -> np.ndarray:
    if "startup_inventory_g" in data_frame.columns:
        values = pd.to_numeric(data_frame["startup_inventory_g"], errors="coerce")
        if "status" in data_frame.columns:
            status_mask = data_frame["status"].astype(str).str.lower() == "success"
            values = values[status_mask]
        return _clean_result_array(values.to_numpy())

    numeric_columns = [
        column
        for column in data_frame.columns
        if pd.api.types.is_numeric_dtype(data_frame[column])
    ]
    numeric_columns = [column for column in numeric_columns if column != "iteration"]

    if len(numeric_columns) == 1:
        return _clean_result_array(data_frame[numeric_columns[0]].to_numpy())

    raise ValueError(
        "CSV file must contain 'startup_inventory_g' or exactly one numeric result column."
    )


def compute_statistics(results: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(results)),
        "std": float(np.std(results)),
        "ci_2_5": float(np.percentile(results, 2.5)),
        "ci_97_5": float(np.percentile(results, 97.5)),
        "min": float(np.min(results)),
        "max": float(np.max(results)),
        "count": int(results.size),
    }


def build_report_text(statistics: Dict[str, float]) -> str:
    return "\n".join(
        [
            "蒙特卡洛不确定性分析报告",
            "=" * 32,
            f"有效样本数: {statistics['count']}",
            f"均值 Mean: {statistics['mean']:.4f} g",
            f"标准差 Standard Deviation: {statistics['std']:.4f} g",
            f"最小值 Min: {statistics['min']:.4f} g",
            f"最大值 Max: {statistics['max']:.4f} g",
            (
                "95%置信区间: "
                f"[{statistics['ci_2_5']:.4f}, {statistics['ci_97_5']:.4f}] g"
            ),
        ]
    )


def save_report(report_text: str, output_path: str) -> None:
    Path(output_path).write_text(report_text, encoding="utf-8")


def plot_results(results: np.ndarray, statistics: Dict[str, float], output_path: str) -> None:
    configure_matplotlib_for_chinese()

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.hist(
        results,
        bins=30,
        density=True,
        alpha=0.78,
        color="#8EC5FC",
        edgecolor="#1F2937",
        linewidth=0.8,
    )

    ax.axvline(
        statistics["mean"],
        color="#C1121F",
        linestyle="--",
        linewidth=2,
        label=f"均值 = {statistics['mean']:.2f} g",
    )
    ax.axvline(
        statistics["ci_2_5"],
        color="#2A9D8F",
        linestyle=":",
        linewidth=2,
        label=f"2.5% = {statistics['ci_2_5']:.2f} g",
    )
    ax.axvline(
        statistics["ci_97_5"],
        color="#2A9D8F",
        linestyle=":",
        linewidth=2,
        label=f"97.5% = {statistics['ci_97_5']:.2f} g",
    )

    stats_text = "\n".join(
        [
            f"样本数: {statistics['count']}",
            f"均值: {statistics['mean']:.2f} g",
            f"标准差: {statistics['std']:.2f} g",
            f"95%区间: [{statistics['ci_2_5']:.2f}, {statistics['ci_97_5']:.2f}] g",
        ]
    )
    ax.text(
        0.98,
        0.97,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )

    ax.set_title("氚启动存量不确定性分析", fontsize=16, pad=12)
    ax.set_xlabel("最小氚启动存量 (g)", fontsize=12)
    ax.set_ylabel("概率密度", fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def analyze_results_file(input_path: str, output_dir: str) -> Dict[str, str]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    results = load_results(input_path)
    statistics = compute_statistics(results)
    report_text = build_report_text(statistics)

    report_path = output_root / "report.txt"
    figure_path = output_root / "uncertainty_analysis.png"

    save_report(report_text, str(report_path))
    plot_results(results, statistics, str(figure_path))

    logger.info(
        "Result analysis completed",
        extra={
            "input_path": str(Path(input_path).resolve()),
            "report_path": str(report_path.resolve()),
            "figure_path": str(figure_path.resolve()),
        },
    )

    return {
        "report": str(report_path.resolve()),
        "figure": str(figure_path.resolve()),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze Monte Carlo startup inventory results and generate a report."
    )
    parser.add_argument("input", type=str, help="Path to results.npy or results.csv")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_output",
        help="Directory used to save report.txt and uncertainty_analysis.png",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level, for example INFO or DEBUG.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    analyze_results_file(args.input, args.output_dir)


if __name__ == "__main__":
    main()