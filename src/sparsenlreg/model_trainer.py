import numpy as np
from sparsenlreg.model import Model
from sklearn.model_selection import train_test_split


from sparsenlreg.feature_manager import FeatureManager


class ModelTrainer:
    def __init__(self, config):
        # Training
        self.log_beta_min = config.log_beta_min
        self.log_beta_max = config.log_beta_max
        self.N_fit = config.N_fit
        self.fit_intercept = config.fit_intercept
        self.test_size = config.test_size
        # Selection
        self.N_coeff_max = config.N_coeff_max
        self.coordinate_descent_selection = config.coordinate_descent_selection
        # Setup the feature manager
        self.feature_manager = FeatureManager(config)

    def _generate_betas(self):
        return 10 ** np.linspace(self.log_beta_min, self.log_beta_max, num=self.N_fit)

    def setup_training_data(self, df, degree: int):
        # Generate the features
        X, X_names = self.feature_manager.generate_polynomial_features(df, degree)
        y, y_name = self.feature_manager.generate_output(df)
        X_scaled = self.feature_manager.scale_features(X)

        # Split the dataset
        # NOTE:
        #   Usually, it is recommend to use a training-validation-test split.
        #   Here, it is not requires as we have no hyperparameter to tweak.
        if self.test_size != 0.0:
            data = train_test_split(
                X_scaled, y, test_size=self.test_size, random_state=42
            )
        else:
            data = X_scaled, X_scaled, y, y
        X_train, X_test, y_train, y_test = data
        return X, X_train, X_test, y, y_train, y_test, X_names, y_name

    def train(self, X_train, y_train, X_test, y_test):
        # Define the values of beta
        betas = self._generate_betas()
        # Create arrays to store the results
        models = [None for _ in range(self.N_fit)]
        # Perform the sparse regression for different values of beta
        for i, beta in enumerate(betas):
            model = Model(
                alpha=beta,
                l1_ratio=1.0,
                fit_intercept=self.fit_intercept,
                tol=1e-6,
                max_iter=512,
                selection=self.coordinate_descent_selection,
            )
            model.fit_and_assess(X_train, y_train, X_test, y_test)
            models[i] = model
        return models

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
