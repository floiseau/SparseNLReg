from functools import partial

import numpy as np
from sparsenlreg.model import LarsModel, ElasticNetModel
from sklearn.model_selection import train_test_split

from sparsenlreg.feature_manager import FeatureManager


class ModelTrainer:
    def __init__(self, config):
        # Training
        self.log_beta_min = config.log_beta_min
        self.log_beta_max = config.log_beta_max
        self.N_fit = config.N_fit
        self.test_size = config.test_size
        self.solver = config.solver
        # Selection
        self.N_coeff_max = config.N_coeff_max
        self.coordinate_descent_selection = config.coordinate_descent_selection
        # Setup the feature manager
        self.feature_manager = FeatureManager(config)

    def create_model(self, beta):
        match self.solver.lower():
            case "lars":
                model = LarsModel(
                    alpha=beta,
                    fit_intercept=False,
                    max_iter=512,
                    # fit_path=False,
                )
            case "coordinate_descent":
                model = ElasticNetModel(
                    alpha=beta,
                    l1_ratio=1.0,
                    fit_intercept=False,
                    tol=1e-6,
                    max_iter=512,
                    selection=self.coordinate_descent_selection,
                )
            case _:
                ValueError(
                    f"The solver {self.solver} is invalid! Value choices are 'lars' or 'coordinate_descent'."
                )

        return model

    def setup_training_data(self, df):
        # Generate the features
        X, X_names = self.feature_manager.generate_expanded_features(df)
        y, y_name = self.feature_manager.generate_output(df)
        X_scaled = self.feature_manager.scale_features(X)
        y_scaled = self.feature_manager.scale_output(y)

        # Split the dataset
        # NOTE:
        #   Usually, it is recommend to use a training-validation-test split.
        #   Here, it is not requires as we have no hyperparameter to tweak.
        if self.test_size != 0.0:
            data = train_test_split(
                X_scaled, y_scaled, test_size=self.test_size, random_state=42
            )
        else:
            data = X_scaled, X_scaled, y_scaled, y_scaled
        X_train, X_test, y_train, y_test = data
        return X, X_train, X_test, y, y_train, y_test, X_names, y_name

    def _generate_betas(self):
        return 10 ** np.linspace(self.log_beta_min, self.log_beta_max, num=self.N_fit)

    def _train_single_model(self, beta, X_train, y_train, X_test, y_test):
        model = self.create_model(beta)
        model.fit_and_assess(X_train, y_train, X_test, y_test)
        return model

    def train(self, X_train, y_train, X_test, y_test):
        # Define the values of beta
        betas = self._generate_betas()
        # Perform the sparse regression for different values of beta
        func = partial(
            self._train_single_model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        return list(map(func, betas))

    def select_best_models(self, models):
        # Create a list of models
        best_models = []
        for N_c in range(1, self.N_coeff_max + 1):
            mask = np.where([m.N_coefs == N_c for m in models])[0]
            if not mask.any():
                continue
            models_with_N_c_coeffs = [models[i] for i in mask]
            id_best = np.argmax(m.r2_score for m in models_with_N_c_coeffs)
            best_models.append(models_with_N_c_coeffs[id_best])
        return best_models
