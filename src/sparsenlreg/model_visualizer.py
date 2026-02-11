import matplotlib.pyplot as plt
from sklearn.metrics import PredictionErrorDisplay


class ModelVisualizer:
    """Handles visualization and export of model plots."""

    def export_tradeoff_plot(self, models, model_name: str):
        """Export the tradeoff plot between accuracy and simplicity."""
        # Extract the plotted quanties
        betas = [model.alpha for model in models]
        r2s = [model.r2_score for model in models]
        Ns_coeff = [model.N_coefs for model in models]

        # Generate the plot
        fig, ax1 = plt.subplots()
        ax1.set_xscale("log")
        ax1.set_xlabel("Hyper-parameter $\\beta$")
        ax1.set_ylabel(
            "Accuracy : Coefficient of determination $R^2$", color="tab:blue"
        )
        ax1.set_ylim((-0.05, 1.05))
        ax1.scatter(betas, r2s, color="tab:blue", marker="o")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Simplicity : Number of coefficients", color="tab:red")
        ax2.scatter(betas, Ns_coeff, color="tab:red", marker="s")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax2.yaxis.get_major_locator().set_params(integer=True)

        plt.title("Accuracy-Simplicity Tradeoff plot")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"{model_name}_tradeoff.svg")
        plt.close()

    def export_actual_vs_predicted_plot(self, best_models, X, y, model_name: str):
        """Export the actual vs. predicted plot."""
        plt.figure()
        ax = plt.gca()
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for m, c in zip(best_models, colors):
            PredictionErrorDisplay.from_estimator(
                m,
                X,
                y,
                kind="actual_vs_predicted",
                ax=ax,
                scatter_kwargs={"color": c, "label": m.label},
            )
        plt.title("Actual vs. Predicted plot")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"{model_name}_actual_vs_predicted.svg")
        plt.close()

    def export_best_models_predictions_plot(
        self, best_models, X, y, X_names, y_name, model_name: str
    ):
        """Export the best models' predictions vs. data plot."""
        plt.figure()
        plt.xlabel(rf"Feature : ${X_names[0]}$")
        plt.ylabel(rf"Output : ${y_name}$")
        # Add the models
        for m in best_models:
            y_pred = m.predict(X)
            plt.plot(X[:, 0], y_pred, label=m.label, ls="--")
        # Add the raw data
        plt.scatter(X[:, 0], y, color="tab:red", marker="o", label="Data")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{model_name}_predictions_vs_data.svg")
        plt.close()

    def export_all_plots(
        self, all_models, best_models, X, y, X_names, y_name, model_name: str
    ):
        """Export all plots at once."""
        self.export_tradeoff_plot(all_models, model_name)
        self.export_actual_vs_predicted_plot(best_models, X, y, model_name)
        self.export_best_models_predictions_plot(
            best_models, X, y, X_names, y_name, model_name
        )
