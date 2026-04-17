import json

import pytest

from tricys.analysis.parameter_sampler import ParameterSampler, generate_sample


def test_default_sampler_generates_expected_keys():
    sampler = ParameterSampler(seed=42)

    sample = sampler.generate_sample()

    assert list(sample.keys()) == [
        "TBR",
        "burn_fraction",
        "tep_residence_time",
        "tes_residence_time",
    ]
    assert 0.0 <= sample["burn_fraction"] <= 1.0
    assert sample["tep_residence_time"] >= 0.0
    assert sample["tes_residence_time"] >= 0.0


def test_sampler_can_load_json_config(tmp_path):
    config_path = tmp_path / "sampler_config.json"
    config = {
        "parameters": {
            "blanket.TBR": {
                "distribution": "norm",
                "mean": 1.08,
                "std": 0.02,
            },
            "plasma.fb": {
                "distribution": "uniform",
                "low": 0.6,
                "high": 0.9,
            },
        }
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    sampler = ParameterSampler(config_path=str(config_path), seed=123)
    sample = sampler.generate_sample()

    assert set(sample.keys()) == {"blanket.TBR", "plasma.fb"}
    assert 0.6 <= sample["plasma.fb"] <= 0.9


def test_sampler_reproducible_with_seed():
    sampler_a = ParameterSampler(seed=7)
    sampler_b = ParameterSampler(seed=7)

    assert sampler_a.generate_sample() == sampler_b.generate_sample()


def test_invalid_distribution_raises_error():
    with pytest.raises(ValueError):
        ParameterSampler(
            parameter_config={
                "parameters": {
                    "bad_param": {
                        "distribution": "gaussianish",
                        "mean": 1.0,
                        "std": 0.1,
                    }
                }
            }
        )


def test_generate_samples_requires_positive_count():
    sampler = ParameterSampler(seed=1)

    with pytest.raises(ValueError):
        sampler.generate_samples(0)


def test_module_level_generate_sample_wrapper():
    sample = generate_sample(seed=9)

    assert "TBR" in sample