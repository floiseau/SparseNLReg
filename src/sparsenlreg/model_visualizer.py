import matplotlib.pyplot as plt
from sklearn.metrics import PredictionErrorDisplay


class ModelVisualizer:
    """Handles visualization and export of model plots."""

    def export_tradeoff_plot(self, betas, r2s, Ns_coeff, filename="tradeoff.svg"):
        """Export the tradeoff plot between accuracy and simplicity."""
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
        plt.savefig(filename)
        plt.close()

    def export_actual_vs_predicted_plot(
        self, best_models, X, y, filename="actual_vs_predicted.svg"
    ):
        """Export the actual vs. predicted plot."""
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
        plt.savefig(filename)
        plt.close()

    def export_best_models_predictions_plot(
        self, best_models, X, y, X_names, y_name, filename="predictions_vs_data.svg"
    ):
        """Export the best models' predictions vs. data plot."""
        plt.figure()
        plt.xlabel(rf"Feature : ${X_names[0]}$")
        plt.ylabel(rf"Output : ${y_name}$")
        # Add the models
        for m in best_models:
            y_pred = m["model"].predict(X)
            plt.plot(X[:, 0], y_pred, label=m["label"], ls="--", marker="x")
        # Add the raw data
        plt.scatter(X[:, 0], y, color="tab:red", marker="o", label="Data")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def export_all_plots(
        self, best_models, X, y, X_names, y_name, betas, r2s, Ns_coeff
    ):
        """Export all plots at once."""
        self.export_tradeoff_plot(betas, r2s, Ns_coeff)
        self.export_actual_vs_predicted_plot(best_models, X, y)
        self.export_best_models_predictions_plot(best_models, X, y, X_names, y_name)
