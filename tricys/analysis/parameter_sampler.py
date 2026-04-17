"""Parameter sampling utilities for uncertainty analysis.

This module provides a lightweight sampler for uncertain input parameters.
It is intentionally independent from the TRICYS execution pipeline so it can
be reused in standalone Monte Carlo scripts or integrated into future analysis
workflows.
"""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

SUPPORTED_DISTRIBUTIONS = {"normal", "norm", "uniform", "unif", "lognormal", "lognorm", "triangular", "triang", "beta"}


DEFAULT_PARAMETER_CONFIG = {
    "parameters": {
        "TBR": {
            "distribution": "normal",
            "mean": 1.1,
            "std": 0.1,
        },
        "burn_fraction": {
            "distribution": "normal",
            "mean": 0.8,
            "std": 0.15,
            "bounds": [0.0, 1.0],
        },
        "tep_residence_time": {
            "distribution": "normal",
            "mean": 2.0,
            "std": 0.3,
            "bounds": [0.0, None],
        },
        "tes_residence_time": {
            "distribution": "normal",
            "mean": 12.0,
            "std": 2.0,
            "bounds": [0.0, None],
        },
    }
}


class ParameterSampler:
    """Sampler for uncertain model parameters.

    The sampler can be constructed from a Python dictionary or a JSON file.
    When no external configuration is provided, it falls back to a built-in
    set of uncertainty definitions suitable for a first Monte Carlo prototype.
    """

    def __init__(
        self,
        parameter_config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        if parameter_config is not None and config_path is not None:
            raise ValueError("Provide either parameter_config or config_path, not both.")

        if config_path is not None:
            loaded_config = self._load_config_from_json(config_path)
        elif parameter_config is not None:
            loaded_config = deepcopy(parameter_config)
        else:
            loaded_config = deepcopy(DEFAULT_PARAMETER_CONFIG)

        self.config = self._normalize_config(loaded_config)
        self._validate_config(self.config)
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _load_config_from_json(config_path: str) -> Dict[str, Any]:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Parameter config file not found: {config_path}")

        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _normalize_config(raw_config: Dict[str, Any]) -> Dict[str, Any]:
        if "parameters" in raw_config:
            return raw_config
        return {"parameters": raw_config}

    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        parameters = config.get("parameters")
        if not isinstance(parameters, dict) or not parameters:
            raise ValueError("Parameter configuration must contain a non-empty 'parameters' dictionary.")

        for parameter_name, parameter_definition in parameters.items():
            if not isinstance(parameter_definition, dict):
                raise ValueError(
                    f"Parameter '{parameter_name}' definition must be a dictionary."
                )

            distribution = parameter_definition.get("distribution")
            if distribution not in SUPPORTED_DISTRIBUTIONS:
                raise ValueError(
                    f"Unsupported distribution '{distribution}' for parameter '{parameter_name}'."
                )

            normalized_distribution = ParameterSampler._normalize_distribution_name(
                distribution
            )

            if normalized_distribution == "normal":
                ParameterSampler._require_keys(
                    parameter_name, parameter_definition, ["mean", "std"]
                )
                if parameter_definition["std"] < 0:
                    raise ValueError(
                        f"Parameter '{parameter_name}' must have a non-negative std."
                    )

            elif normalized_distribution == "uniform":
                ParameterSampler._require_keys(
                    parameter_name, parameter_definition, ["low", "high"]
                )
                if parameter_definition["low"] >= parameter_definition["high"]:
                    raise ValueError(
                        f"Parameter '{parameter_name}' must satisfy low < high."
                    )

            elif normalized_distribution == "lognormal":
                ParameterSampler._require_keys(
                    parameter_name, parameter_definition, ["mean", "sigma"]
                )
                if parameter_definition["sigma"] < 0:
                    raise ValueError(
                        f"Parameter '{parameter_name}' must have a non-negative sigma."
                    )

            elif normalized_distribution == "triangular":
                ParameterSampler._require_keys(
                    parameter_name, parameter_definition, ["low", "mode", "high"]
                )
                low = parameter_definition["low"]
                mode = parameter_definition["mode"]
                high = parameter_definition["high"]
                if not low <= mode <= high:
                    raise ValueError(
                        f"Parameter '{parameter_name}' must satisfy low <= mode <= high."
                    )

            elif normalized_distribution == "beta":
                ParameterSampler._require_keys(
                    parameter_name, parameter_definition, ["alpha", "beta"]
                )
                if parameter_definition["alpha"] <= 0 or parameter_definition["beta"] <= 0:
                    raise ValueError(
                        f"Parameter '{parameter_name}' must have positive alpha and beta."
                    )

    @staticmethod
    def _require_keys(
        parameter_name: str,
        parameter_definition: Dict[str, Any],
        required_keys: List[str],
    ) -> None:
        missing_keys = [key for key in required_keys if key not in parameter_definition]
        if missing_keys:
            raise ValueError(
                f"Parameter '{parameter_name}' is missing required keys: {missing_keys}"
            )

    @staticmethod
    def _normalize_distribution_name(distribution: str) -> str:
        alias_map = {
            "norm": "normal",
            "unif": "uniform",
            "lognorm": "lognormal",
            "triang": "triangular",
        }
        return alias_map.get(distribution, distribution)

    @staticmethod
    def _apply_bounds(value: float, bounds: Optional[List[Optional[float]]]) -> float:
        if not bounds:
            return float(value)

        lower_bound, upper_bound = bounds
        if lower_bound is not None:
            value = max(value, lower_bound)
        if upper_bound is not None:
            value = min(value, upper_bound)
        return float(value)

    def get_parameter_names(self) -> List[str]:
        """Return all parameter names in configuration order."""
        return list(self.config["parameters"].keys())

    def generate_sample(self) -> Dict[str, float]:
        """Generate one random sample for all configured parameters."""
        sample = {}
        for parameter_name, parameter_definition in self.config["parameters"].items():
            sample[parameter_name] = self._sample_parameter(parameter_definition)
        return sample

    def generate_samples(self, count: int) -> List[Dict[str, float]]:
        """Generate multiple random samples."""
        if count <= 0:
            raise ValueError("Sample count must be a positive integer.")
        return [self.generate_sample() for _ in range(count)]

    def _sample_parameter(self, parameter_definition: Dict[str, Any]) -> float:
        distribution = self._normalize_distribution_name(
            parameter_definition["distribution"]
        )

        if distribution == "normal":
            value = self.rng.normal(
                loc=parameter_definition["mean"],
                scale=parameter_definition["std"],
            )
        elif distribution == "uniform":
            value = self.rng.uniform(
                low=parameter_definition["low"],
                high=parameter_definition["high"],
            )
        elif distribution == "lognormal":
            value = self.rng.lognormal(
                mean=parameter_definition["mean"],
                sigma=parameter_definition["sigma"],
            )
        elif distribution == "triangular":
            value = self.rng.triangular(
                left=parameter_definition["low"],
                mode=parameter_definition["mode"],
                right=parameter_definition["high"],
            )
        elif distribution == "beta":
            value = self.rng.beta(
                a=parameter_definition["alpha"],
                b=parameter_definition["beta"],
            )
            if "low" in parameter_definition or "high" in parameter_definition:
                low = parameter_definition.get("low", 0.0)
                high = parameter_definition.get("high", 1.0)
                value = low + value * (high - low)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

        return self._apply_bounds(value, parameter_definition.get("bounds"))


def generate_sample(
    parameter_config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Convenience wrapper that returns one sampled parameter set."""
    sampler = ParameterSampler(
        parameter_config=parameter_config,
        config_path=config_path,
        seed=seed,
    )
    return sampler.generate_sample()


__all__ = ["DEFAULT_PARAMETER_CONFIG", "ParameterSampler", "generate_sample"]