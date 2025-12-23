import tomllib
from dataclasses import dataclass


@dataclass(kw_only=True)
class LASSOConfig:
    # Dataset
    dataset: str

    # Model features
    base_features: list
    base_feature_names: list

    # Model output
    output: str
    output_name: str

    # Polynomial parameters
    degree: int
    fit_intercept: bool

    # Tradeoff exploration
    N_fit: int
    log_beta_min: float
    log_beta_max: float

    # Model selection
    N_coeff_max: int
    test_size: float

    # Training
    polish: bool


def parse_config_file(file_path: str) -> LASSOConfig:
    """Parse the TOML config file and return the LASSOConfig."""
    with open(file_path, "rb") as f:
        config = tomllib.load(f)

    # Flatten the nested dictionary structure
    flattened_config = {
        "dataset": config["data"]["dataset"],
        "base_features": config["model_features"]["base_features"],
        "base_feature_names": config["model_features"]["base_feature_names"],
        "output": config["model_output"]["output"],
        "output_name": config["model_output"]["output_name"],
        "degree": config["polynomial"]["degree"],
        "fit_intercept": config["polynomial"]["fit_intercept"],
        "N_fit": config["tradeoff_exploration"]["N_fit"],
        "log_beta_min": config["tradeoff_exploration"]["log_beta_min"],
        "log_beta_max": config["tradeoff_exploration"]["log_beta_max"],
        "N_coeff_max": config["model_selection"]["N_coeff_max"],
        "test_size": config["model_selection"].get("test_size", 0.0),
        "polish": config.get("training", {}).get("polish", False),
    }

    return LASSOConfig(**flattened_config)
