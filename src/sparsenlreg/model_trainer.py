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
                max_iter=1_000,
            )
            model.fit(X, y)
            models[i] = model
        return models, betas

    def assess_models_scores(self, models, X, y):
        return np.array([r2_score(y, m.predict(X)) for m in models])

    def count_nonzero_coefficients(self, models):
        return np.array(
            [np.count_nonzero(m.coef_) + self.fit_intercept for m in models]
        )

    def select_best_models(self, models, betas, r2s, Ns_coeffs):
        # Create a list of models
        best_models = []
        for N_c in range(0, self.N_coeff_max + 1):
            mask = Ns_coeffs == N_c
            if not mask.any():
                continue
            id_best = np.argmax(r2s[mask])
            best_r2 = r2s[mask][id_best]
            if best_r2 < 0:
                continue
            model = np.array(models)[mask][id_best]
            best_models.append(
                {
                    "model": model,
                    "r2": r2s[mask][id_best],
                    "N_coeff": N_c,
                    "beta": betas[mask][id_best],
                    "label": f"Model : $N_\\mathrm{{coeff}} = {N_c}$, $R^2 = {r2s[mask][id_best]:.3f}$",
                }
            )

        return best_models
