import argparse
import tomllib
from typing import List

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


def generate_base_features(
    df: pd.DataFrame,
    features: List[str],
):
    # Create the list of features from observations and non-linear terms
    X_base_feat_list = []
    for feat in features:
        if feat in df.columns:
            X_base_feat_list.append(df[feat])
        else:
            # Convert the feature to a SymPy expression
            feat_expr = sp.sympify(feat)
            # Exctract the symbols
            symbols = list(feat_expr.free_symbols)
            symbol_names = [symb.name for symb in symbols]
            # Convert to the SymPy expression to a numerical function
            f = sp.lambdify(symbols, feat_expr, "numpy")
            # Evaluate the function
            f_args = (df[name] for name in symbol_names)
            X_base_feat_list.append(f(*f_args))
    # Gather the features into an array
    X_base_feat = np.column_stack(X_base_feat_list)
    return X_base_feat


def transform_base_features_to_polynomial_features(
    X_base_feat,
    X_base_feat_names,
    degree=10,
    include_bias=True,
):
    """
    Convert an array of base features into an array of polynomial features.

    Parameters:
    - X_base_feat: array-like, input features
    - X_base_feat_names: array-like, names of input features
    - degree: int, degree of polynomial features (default: 10)
    - include_bias: bool, add a constant term to the polynomial if True

    Returns:
    - X: ndarray, matrix of polynomial features
    - X_names: list, names of the polynomial features
    """
    # Generate polynomial features
    pf = PolynomialFeatures(degree=degree, include_bias=True)
    X = pf.fit_transform(X_base_feat)
    X_names = pf.get_feature_names_out(X_base_feat_names)
    return X, X_names


def perform_fit_for_varying_beta_to_assess_tradeoff(X, y, betas):
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
            alpha=beta, fit_intercept=True, tol=1e-6, max_iter=1_000
        )
        _model.fit(X, y)
        models[i] = _model
    return models


def plot_tradeoff(betas, r2s, Ns_coeff):
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
    plt.savefig("tradeoff_plot.svg")


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


def generate_latex_table(best_models, X_names, y_name):
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

    return latex_output


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
    include_bias = config["polynomial"]["include_bias"]
    # Tradeoff exploration
    N_fit = config["tradeoff_exploration"]["N_fit"]
    log_beta_min = config["tradeoff_exploration"]["log_beta_min"]
    log_beta_max = config["tradeoff_exploration"]["log_beta_max"]
    # Model selection
    N_coeff_max = config["model_selection"]["N_coeff_max"]

    # Read the CSV file
    df = pd.read_csv(config["data"]["dataset"], comment="#")
    # Create the output vector
    y = df[output].to_numpy()
    y_name = output_name
    # Generate the features
    X_base_feat = generate_base_features(df, base_features)
    X, X_names = transform_base_features_to_polynomial_features(
        X_base_feat,
        base_feature_names,
        degree=degree,
        include_bias=include_bias,
    )

    # Fit for different values of hyper-parameter beta to assess the tradeoff
    betas = 10 ** np.linspace(log_beta_min, log_beta_max, num=N_fit)
    models = perform_fit_for_varying_beta_to_assess_tradeoff(X, y, betas)
    r2s = np.array([r2_score(y, m.predict(X)) for m in models])
    Ns_coeffs = np.array([np.count_nonzero(m.coef_) + include_bias for m in models])

    # Display the tradeoff plot
    plot_tradeoff(betas, r2s, Ns_coeffs)

    # Gather the best models per number of coefficients in a dictionary
    best_models = gather_best_models(models, betas, r2s, Ns_coeffs, N_coeff_max)
    # Display the best models
    display_best_models(best_models, X_names, y_name)
    latex_table = generate_latex_table(best_models, X_names, y_name)
    print(latex_table)

    # Initialize the figure
    plt.figure()
    plt.xlabel(rf"Feature : ${base_feature_names[0]}$")
    plt.ylabel(rf"Output : ${y_name}$")
    # Add the models
    for m in best_models:
        y_pred = m["model"].predict(X)
        label = f"Model : $R^2 = {m['r2']:.3f}$, $N_\\mathrm{{coeff}} = {m['N_coeff']}$"
        plt.plot(X_base_feat[:, 0], y_pred, label=label, ls="--", marker="x")
    # Add the raw data
    plt.scatter(X_base_feat[:, 0], y, color="tab:red", marker="o", label="Data")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("best_models_comparison_physics.svg")

    # Display te actual vs predicted plot
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
            scatter_kwargs={"color": c},
        )
    plt.grid()
    plt.savefig("best_models_comparison_data_science.svg")


if __name__ == "__main__":
    main()
