import argparse
import tomllib
from typing import List, Union

import pandas as pd
import sympy as sp
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, PredictionErrorDisplay
import matplotlib.pyplot as plt


def parse_config(file_path: str) -> dict:
    """Parse the TOML config file and return a dictionary."""
    with open(file_path, "rb") as f:
        config = tomllib.load(f)
    return config


def _evaluate_expression_or_column(
    df: pd.DataFrame,
    expr: Union[str, List[str]],
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Helper function to evaluate either a column or a SymPy expression.
    If expr is a list, returns a list of arrays; otherwise, returns a single array.
    """
    is_list = isinstance(expr, list)
    exprs = expr if is_list else [expr]

    results = []
    for e in exprs:
        if e in df.columns:
            results.append(df[e].to_numpy())
        else:
            expr_sym = sp.sympify(e)
            symbols = list(expr_sym.free_symbols)
            symbol_names = [symb.name for symb in symbols]
            f = sp.lambdify(symbols, expr_sym, "numpy")
            f_args = (df[name] for name in symbol_names)
            results.append(f(*f_args))

    return results if is_list else results[0]


def generate_base_features(
    df: pd.DataFrame,
    features: List[str],
) -> np.ndarray:
    """
    Generate base features from a DataFrame and a list of feature expressions.
    """
    X_base_feat_list = _evaluate_expression_or_column(df, features)
    return np.column_stack(X_base_feat_list)


def generate_output(
    df: pd.DataFrame,
    output: str,
) -> np.ndarray:
    """
    Generate output from a DataFrame and an output expression.
    """
    return _evaluate_expression_or_column(df, output)


def transform_base_features_to_polynomial_features(
    X_base_feat,
    X_base_feat_names,
    degree=10,
):
    """
    Convert an array of base features into an array of polynomial features.

    Parameters:
    - X_base_feat: array-like, input features
    - X_base_feat_names: array-like, names of input features
    - degree: int, degree of polynomial features (default: 10)

    Returns:
    - X: ndarray, matrix of polynomial features
    - X_names: list, names of the polynomial features
    """
    # Generate polynomial features
    pf = PolynomialFeatures(degree=degree, include_bias=False)
    X = pf.fit_transform(X_base_feat)
    X_names = pf.get_feature_names_out(X_base_feat_names)
    return X, X_names


def perform_fit_for_varying_beta_to_assess_tradeoff(X, y, betas, fit_intercept):
    """
    Perform Lasso regression for a range of beta values.

    Parameters:
    -----------
    X : array-like
        Input features.
    y : array-like
        Expected output values.
    bets : int, optional
        Number of beta values to fit (default: 128).
    N_coeff_max : int, optional
        Maximum number of coefficients to consider for best models (default: 8).
    plot : bool, optional
        Whether to plot the tradeoff (default: True).

    Returns:
    --------
    TODO
    """
    # Create arrays to store the results
    models = [None for _ in range(len(betas))]
    # Perform the sparse regression for different values of beta
    for i, beta in enumerate(betas):
        _model = linear_model.Lasso(
            alpha=beta, fit_intercept=fit_intercept, tol=1e-6, max_iter=1_000
        )
        _model.fit(X, y)
        models[i] = _model
    return models


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


def gather_best_models(models, betas, r2s, Ns_coeffs, N_coeff_max: int):
    best_models = []
    for N_c in range(0, N_coeff_max + 1):
        mask = Ns_coeffs == N_c
        if not mask.any():
            continue
        id_best = np.argmax(r2s[mask])
        best_r2 = r2s[mask][id_best]
        if best_r2 < 0:
            continue
        _model = np.array(models)[mask][id_best]
        best_models.append(
            {
                "model": _model,
                "r2": r2s[mask][id_best],
                "N_coeff": N_c,
                "beta": betas[mask][id_best],
                "label": f"Model : $N_\\mathrm{{coeff}} = {N_c}$, $R^2 = {r2s[mask][id_best]:.3f}$",
            }
        )

    return best_models


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
        print(f"{y_name}(a) =\n\t{cst:+.3g}", end="\n")
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
    plt.legend()
    plt.grid()
    plt.savefig("actual_vs_predicted.svg")


def export_best_models_predictions_plot(
    best_models, X_base_feat, X, y, base_feature_names, y_name
):
    plt.figure()
    plt.xlabel(rf"Feature : ${base_feature_names[0]}$")
    plt.ylabel(rf"Output : ${y_name}$")
    # Add the models
    for m in best_models:
        y_pred = m["model"].predict(X)
        plt.plot(X_base_feat[:, 0], y_pred, label=m["label"], ls="--", marker="x")
    # Add the raw data
    plt.scatter(X_base_feat[:, 0], y, color="tab:red", marker="o", label="Data")
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
    parser = argparse.ArgumentParser(
        prog="sparsenlreg",
        description="TODO",
        usage="sparsenlreg configuration.toml",
    )
    parser.add_argument("config_file", type=str)
    args = parser.parse_args()
    print(f"Reading the configuration file : {args.config_file}.")

    # Read the configuration file
    config = parse_config(args.config_file)
    # Model features
    base_features = config["model_features"]["base_features"]
    base_feature_names = config["model_features"]["base_feature_names"]
    # Model output
    output = config["model_output"]["output"]
    output_name = config["model_output"]["output_name"]
    # Polynomial parameters
    degree = config["polynomial"]["degree"]
    fit_intercept = config["polynomial"]["fit_intercept"]
    # Tradeoff exploration
    N_fit = config["tradeoff_exploration"]["N_fit"]
    log_beta_min = config["tradeoff_exploration"]["log_beta_min"]
    log_beta_max = config["tradeoff_exploration"]["log_beta_max"]
    # Model selection
    N_coeff_max = config["model_selection"]["N_coeff_max"]

    # Read the CSV file
    df = pd.read_csv(config["data"]["dataset"], comment="#")
    # Generate the features
    X_base_feat = generate_base_features(df, base_features)
    X, X_names = transform_base_features_to_polynomial_features(
        X_base_feat,
        base_feature_names,
        degree=degree,
    )
    # Create the output vector
    y = generate_output(df, output)
    y_name = output_name

    # Fit for different values of hyper-parameter beta to assess the tradeoff
    betas = 10 ** np.linspace(log_beta_min, log_beta_max, num=N_fit)
    models = perform_fit_for_varying_beta_to_assess_tradeoff(X, y, betas, fit_intercept)
    r2s = np.array([r2_score(y, m.predict(X)) for m in models])
    Ns_coeffs = np.array([np.count_nonzero(m.coef_) + fit_intercept for m in models])

    # Gather the best models per number of coefficients in a dictionary
    best_models = gather_best_models(models, betas, r2s, Ns_coeffs, N_coeff_max)
    # Display the best models
    display_best_models(best_models, X_names, y_name)
    display_latex_summary(best_models, X_names, y_name)

    # Plots
    export_tradeoff_plot(betas, r2s, Ns_coeffs)
    export_actual_vs_predicted_plot(best_models, X, y)
    export_best_models_predictions_plot(
        best_models, X_base_feat, X, y, base_feature_names, y_name
    )


if __name__ == "__main__":
    main()
