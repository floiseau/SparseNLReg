from pathlib import Path
from datetime import datetime
import argparse

import pandas as pd

from sparsenlreg.configuration import parse_config_file
from sparsenlreg.model_trainer import ModelTrainer
from sparsenlreg.model_reporter import ModelReporter
from sparsenlreg.model_visualizer import ModelVisualizer


def main():
    # NOTE: Vocabulary
    #   Base features: Inputs of the model, composed raw observations and non-polynomial terms.
    #   Polynomial features: Polynomial combinations of the base features. Those are linearly combined to form the model.
    #   Output: raw input from the data that serves as output of the model
    #   Coefficients: Coefficient of the fitted polynomial

    # Create the CLI argument parser
    parser = argparse.ArgumentParser(
        prog="sparsenlreg",
        description="TODO",
        usage="sparsenlreg configuration.toml",
    )
    parser.add_argument("config_file", type=Path)
    args = parser.parse_args()
    print(f"Reading the configuration file : {args.config_file}.")

    # Create a export directory
    model_name = args.config_file.stem
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    export_dir = Path(f"{model_name}_{date}")
    export_dir.mkdir()
    # Copy the configuration file into the export directory
    args.config_file.copy(export_dir / args.config_file.name)

    # Read the configuration file
    config = parse_config_file(args.config_file)
    # Read the CSV file
    df = pd.read_csv(config.dataset, comment="#")

    # Train the models
    model_trainer = ModelTrainer(config)
    X, X_train, X_test, y, y_train, y_test, X_names, y_name = (
        model_trainer.setup_training_data(df)
    )
    models = model_trainer.train(X_train, y_train, X_test, y_test)

    # Refit the models without regularization
    if config.polish:
        for model in models:
            model.refit_model_without_regularization(X_train, y_train, X_test, y_test)

    # Select the best models
    best_models = model_trainer.select_best_models(models)

    # Rescale the best models for plotting
    best_models = model_trainer.feature_manager.rescale_best_models(best_models)

    # Display the model and generate a report
    model_reporter = ModelReporter()
    model_reporter.generate_markdown_report(best_models, X_names, y_name, export_dir)
    model_reporter.display_terminal_table(best_models, X_names, y_name)

    # Export the plots
    visualizer = ModelVisualizer()
    if config.export_plots:
        visualizer = ModelVisualizer()
        visualizer.export_all_plots(
            models, best_models, X, y, X_names, y_name, export_dir
        )
    # Export the data
    if config.export_tradeoff_data:
        visualizer.export_tradeoff_data(models, export_dir)


if __name__ == "__main__":
    main()
