import numpy as np
from sklearn import linear_model


class ModelTrainer:
    def __init__(self, config):
        self.log_beta_min = config.log_beta_min
        self.log_beta_max = config.log_beta_max
        self.N_fit = config.N_fit
        self.fit_intercept = config.fit_intercept

    def train(self, X, y):
        # Define the values of beta
        self.betas = 10 ** np.linspace(
            self.log_beta_min, self.log_beta_max, num=self.N_fit
        )
        # Create arrays to store the results
        models = [None for _ in range(self.N_fit)]
        # Perform the sparse regression for different values of beta
        for i, beta in enumerate(self.betas):
            model = linear_model.Lasso(
                alpha=beta,
                fit_intercept=self.fit_intercept,
                tol=1e-6,
                max_iter=1_000,
            )
            model.fit(X, y)
            models[i] = model
        return models
