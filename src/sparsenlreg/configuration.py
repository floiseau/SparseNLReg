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

    # Expansion
    expansions: list

    # Tradeoff exploration
    N_fit: int
    log_beta_min: float
    log_beta_max: float
    solver: str
    coordinate_descent_selection: str

    # Model selection
    N_coeff_max: int
    test_size: float

    # Training
    polish: bool

    # Export
    export_plots: bool
    export_tradeoff_data: bool


def parse_config_file(file_path: str) -> LASSOConfig:
    """Parse the TOML config file and return the LASSOConfig."""
    with open(file_path, "rb") as f:
        config = tomllib.load(f)

    # Flatten the nested dictionary structure
    flattened_config = {}
    # Dataset
    flattened_config["dataset"] = config["data"]["dataset"]
    # Model Features
    flattened_config["base_features"] = config["model_features"]["base_features"]
    flattened_config["base_feature_names"] = config["model_features"][
        "base_feature_names"
    ]
    # Model Output
    flattened_config["output"] = config["model_output"]["output"]
    flattened_config["output_name"] = config["model_output"]["output_name"]
    # Expansion
    flattened_config["expansions"] = config.get("expansions", [])
    # Tradeoff exploration
    flattened_config["N_fit"] = config["tradeoff_exploration"]["N_fit"]
    flattened_config["log_beta_min"] = config["tradeoff_exploration"]["log_beta_min"]
    flattened_config["log_beta_max"] = config["tradeoff_exploration"]["log_beta_max"]
    flattened_config["solver"] = config["tradeoff_exploration"].get("solver", "lars")
    flattened_config["coordinate_descent_selection"] = config[
        "tradeoff_exploration"
    ].get("coordinate_descent_selection", "cyclic")
    # Model selection
    flattened_config["N_coeff_max"] = config["model_selection"]["N_coeff_max"]
    flattened_config["test_size"] = config["model_selection"].get("test_size", 0.0)
    # Training
    flattened_config["polish"] = config.get("training", {}).get("polish", False)
    # Export
    flattened_config["export_plots"] = config.get("export", {}).get(
        "export_plots", True
    )
    flattened_config["export_tradeoff_data"] = config.get("export", {}).get(
        "export_tradeoff_data", True
    )

    # Check for invalid values
    if flattened_config["coordinate_descent_selection"] not in ["random", "cyclic"]:
        raise ValueError(
            "The coordinate_descent_selection value is invalid. Please choose either 'random' or 'cyclic'."
        )

    return LASSOConfig(**flattened_config)
