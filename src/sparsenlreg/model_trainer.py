import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score


class ModelTrainer:
    def __init__(self, config):
        # Training
        self.log_beta_min = config.log_beta_min
        self.log_beta_max = config.log_beta_max
        self.N_fit = config.N_fit
        self.fit_intercept = config.fit_intercept
        # Selection
        self.N_coeff_max = config.N_coeff_max

    def _generate_betas(self):
        return 10 ** np.linspace(self.log_beta_min, self.log_beta_max, num=self.N_fit)

    def train(self, X, y):
        # Define the values of beta
        betas = self._generate_betas()
        # Create arrays to store the results
        models = [None for _ in range(self.N_fit)]
        # Perform the sparse regression for different values of beta
        for i, beta in enumerate(betas):
            model = linear_model.Lasso(
                alpha=beta,
                fit_intercept=self.fit_intercept,
                tol=1e-6,
                max_iter=1_024,
                selection="cyclic",  # cyclic or random
            )
            model.fit(X, y)
            models[i] = model
        return models, betas

    def assess_model_score(self, model, X, y):
        return r2_score(y, model.predict(X))

    def assess_models_scores(self, models, X, y):
        return np.array([self.assess_model_score(m, X, y) for m in models])

    def count_nonzero_coefficients(self, models):
        return np.array(
            [np.count_nonzero(m.coef_) + self.fit_intercept for m in models]
        )

    def select_best_models(self, models, betas, r2s, Ns_coeffs):
        # Create a list of models
        best_models = []
        for N_c in range(1, self.N_coeff_max + 1):
            mask = Ns_coeffs == N_c
            if not mask.any():
                continue
            id_best = np.argmax(r2s[mask])
            best_models.append(
                {
                    "model": np.array(models)[mask][id_best],
                    "r2": r2s[mask][id_best],
                    "N_coeff": N_c,
                    "beta": betas[mask][id_best],
                    "label": f"Model : $N_\\mathrm{{coeff}} = {N_c}$, $R^2 = {r2s[mask][id_best]:.3f}$",
                }
            )

        return best_models

    def refit_best_models_without_regularization(self, best_models, X, y):
        # Copy the best models list
        model_dicts = best_models.copy()
        # Iterate through the models
        for model_dict in model_dicts:
            # Extract the model
            model = model_dict["model"]
            # Refit the model
            model_dict["model"] = self.refit_model_without_regularization(model, X, y)
            # Update the r2 score
            model_dict["r2_score"] = self.assess_model_score(model, X, y)
        # Return the best models
        return model_dicts

    def refit_model_without_regularization(self, model, X, y):
        """
        Refits the model using only the nonzero coefficients from the LASSO model,
        without L1-regularization.

        This function takes a fitted LASSO model, identifies the nonzero coefficients,
        and refits a linear model without regularization using only those features.
        The refitted coefficients are then stored back in the original LASSO model object.

        Args:
            model: Fitted LASSO model.
            X: Input features.
            y: Target values.

        Returns:
            model: The original LASSO model object with coefficients refitted
                   without regularization. The model's coef_ and intercept_
                   attributes are updated.
        """
        # Get the indices of the nonzero coefficients
        nonzero_indices = np.nonzero(model.coef_)[0]

        if len(nonzero_indices) == 0:
            raise ValueError("No nonzero coefficients found in the model.")

        # Select the features corresponding to nonzero coefficients
        X_selected = X[:, nonzero_indices]

        # Refitting without regularization (using Ordinary Least Squares)
        refitted_linear_model = linear_model.LinearRegression(
            fit_intercept=model.fit_intercept
        )
        refitted_linear_model.fit(X_selected, y)

        # Update the coefficients of the original model
        model.coef_[:] = 0
        model.coef_[nonzero_indices] = refitted_linear_model.coef_

        if model.fit_intercept:
            model.intercept_ = refitted_linear_model.intercept_

        return model
