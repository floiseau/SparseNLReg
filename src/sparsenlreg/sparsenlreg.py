import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from sparsenlreg.configuration import parse_config_file
from sparsenlreg.feature_manager import FeatureManager
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
    parser.add_argument("config_file", type=str)
    args = parser.parse_args()
    print(f"Reading the configuration file : {args.config_file}.")

    # Read the configuration file
    config = parse_config_file(args.config_file)
    # Read the CSV file
    df = pd.read_csv(config.dataset, comment="#")

    # Generate the features
    feature_manager = FeatureManager(config)
    X, X_names = feature_manager.generate_polynomial_features(df)
    y, y_name = feature_manager.generate_output(df)
    X_scaled = feature_manager.scale_features(X)

    #   4. Rescale the parameters with the features
    #       -> https://stackoverflow.com/questions/64668082/how-linear-regression-coefficients-are-stored-in-sklearn-pipelines/64671638#64671638
    #   5. Put everything in the FeatureManager ?

    # Split the dataset
    # NOTE:
    #   Usually, it is recommend to use a training-validation-test split.
    #   Here, it is not requires as we have no hyperparameter to tweak.
    if config.test_size != 0.0:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=config.test_size, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = X_scaled, X_scaled, y, y

    # Train the models
    model_trainer = ModelTrainer(config)
    models, betas = model_trainer.train(X_train, y_train)
    r2s = model_trainer.assess_models_scores(models, X_test, y_test)
    Ns_coeffs = model_trainer.count_nonzero_coefficients(models)
    best_models = model_trainer.select_best_models(models, betas, r2s, Ns_coeffs)

    # Refit the best models without regularization
    if config.polish:
        best_models = model_trainer.refit_best_models_without_regularization(
            best_models, X_train, y_train
        )

    # Rescale the best models for plotting
    best_models = feature_manager.rescale_best_models(best_models)

    # Display the model and generate a report
    model_reporter = ModelReporter()
    model_reporter.generate_markdown_report(best_models, X_names, y_name)
    model_reporter.display_terminal_table(best_models, X_names, y_name)

    # Plots
    visualizer = ModelVisualizer()
    visualizer.export_all_plots(
        best_models, X, y, X_names, y_name, betas, r2s, Ns_coeffs
    )


if __name__ == "__main__":
    main()
