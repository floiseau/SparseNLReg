from pathlib import Path
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

    # Extract the name of the model
    model_name = args.config_file.stem

    # Read the configuration file
    config = parse_config_file(args.config_file)
    # Read the CSV file
    df = pd.read_csv(config.dataset, comment="#")

    # TODO: Change of polynomial order to a list of orders ?

    # Train the models
    model_trainer = ModelTrainer(config)
    X, X_train, X_test, y, y_train, y_test, X_names, y_name = (
        model_trainer.setup_training_data(df, config.degree)
    )
    models = model_trainer.train(X_train, y_train, X_test, y_test)

    # Select the best models
    best_models = model_trainer.select_best_models(models)

    # Refit the best models without regularization
    if config.polish:
        for model in best_models:
            model.refit_model_without_regularization(X_train, y_train, X_test, y_test)

    # Rescale the best models for plotting
    best_models = model_trainer.feature_manager.rescale_best_models(best_models)

    # Display the model and generate a report
    model_reporter = ModelReporter()
    model_reporter.generate_markdown_report(best_models, X_names, y_name, model_name)
    model_reporter.display_terminal_table(best_models, X_names, y_name)

    # Plots
    visualizer = ModelVisualizer()
    visualizer.export_all_plots(models, best_models, X, y, X_names, y_name, model_name)


if __name__ == "__main__":
    main()
