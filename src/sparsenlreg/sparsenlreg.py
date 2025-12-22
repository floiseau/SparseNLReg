import argparse

import pandas as pd
from sklearn.metrics import PredictionErrorDisplay
import matplotlib.pyplot as plt

from sparsenlreg.configuration import parse_config_file
from sparsenlreg.feature_manager import FeatureManager
from sparsenlreg.model_trainer import ModelTrainer


def export_tradeoff_plot(betas, r2s, Ns_coeff):
    fig, ax1 = plt.subplots()
    ax1.set_xscale("log")
    ax1.set_xlabel("Hyper-parameter $\\beta$")
    ax1.set_ylabel("Accuracy : Coefficient of determination $R^2$", color="tab:blue")
    ax1.set_ylim((-0.05, 1.05))
    ax1.scatter(betas, r2s, color="tab:blue", marker="o")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Simplicity : Number of coefficients", color="tab:red")
    ax2.scatter(betas, Ns_coeff, color="tab:red", marker="s")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.yaxis.get_major_locator().set_params(integer=True)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.title("Accuracy-Simplicity Tradeoff plot")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("tradeoff.svg")


def display_best_models(best_models, X_names, y_name="f(X)"):
    for _model_data in best_models:
        cst = _model_data["model"].intercept_
        coeffs = _model_data["model"].coef_
        N_coeff = _model_data["N_coeff"]
        r2 = _model_data["r2"]
        beta = _model_data["beta"]

        print(f"=== Model {N_coeff}")
        print("Assessment :")
        print(f"\t- Hyperparameter beta    : {beta:1.2e}")
        print(f"\t- Number of coefficients : {N_coeff}")
        print(f"\t- R2 score               : {r2:.3f}")

        # Print the model in LaTeX form
        print("LaTeX expression:")
        print(f"{y_name} =\n\t{cst:+.3g}", end="\n")
        for i, coeff in enumerate(coeffs):
            if not coeff == 0:
                print(f"\t{coeff:+.3g} {X_names[i]}", end="\n")
        print("")


def display_latex_summary(best_models, X_names, y_name):
    latex_table = []
    for _model_data in best_models:
        cst = _model_data["model"].intercept_
        coeffs = _model_data["model"].coef_
        N_coeff = _model_data["N_coeff"]
        r2 = _model_data["r2"]
        beta = _model_data["beta"]

        # Build the LaTeX expression for the model
        latex_expr = f"{y_name}(a) = {cst:+.3g}"
        for i, coeff in enumerate(coeffs):
            if not coeff == 0:
                latex_expr += f" {coeff:+.3g} {X_names[i]}"

        # Append the row to the table
        latex_table.append(f"{r2:.3f} & {N_coeff} & {beta:.2g} & ${latex_expr}$ \\\\")

    # Generate the LaTeX table
    latex_output = (
        "\\begin{tabular}{|c|c|c|l|}\n"
        "\\hline\n"
        "$R^2$ & $N_{\\mathrm{coeff}}$ & $\\beta$ & Model \\\\\n"
        "\\hline\n"
    )

    latex_output += "\n".join(latex_table)

    latex_output += "\\\\\n\\hline\n\\end{tabular}\n"

    print("=== Summary in a LaTeX table")
    print(latex_output)


def export_actual_vs_predicted_plot(best_models, X, y):
    plt.figure()
    ax = plt.gca()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for m, c in zip(best_models, colors):
        PredictionErrorDisplay.from_estimator(
            m["model"],
            X,
            y,
            kind="actual_vs_predicted",
            ax=ax,
            scatter_kwargs={"color": c, "label": m["label"]},
        )
    plt.title("Actual vs. Predicted plot")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.svg")


def export_best_models_predictions_plot(best_models, X, y, X_names, y_name):
    plt.figure()
    plt.xlabel(rf"Feature : ${X_names[1]}$")
    plt.ylabel(rf"Output : ${y_name}$")
    # Add the models
    for m in best_models:
        y_pred = m["model"].predict(X)
        plt.plot(X[:, 1], y_pred, label=m["label"], ls="--", marker="x")
    # Add the raw data
    plt.scatter(X[:, 1], y, color="tab:red", marker="o", label="Data")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("predictions_vs_data.svg")


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

    # Train the models
    model_trainer = ModelTrainer(config)
    models, betas = model_trainer.train(X, y)
    r2s = model_trainer.assess_models_scores(models, X, y)
    Ns_coeffs = model_trainer.count_nonzero_coefficients(models)
    best_models = model_trainer.select_best_models(models, betas, r2s, Ns_coeffs)

    # Display the best models
    display_best_models(best_models, X_names, y_name)
    display_latex_summary(best_models, X_names, y_name)

    # Plots
    export_tradeoff_plot(betas, r2s, Ns_coeffs)
    export_actual_vs_predicted_plot(best_models, X, y)
    export_best_models_predictions_plot(best_models, X, y, X_names, y_name)


if __name__ == "__main__":
    main()
