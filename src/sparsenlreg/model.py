import numpy as np

from sklearn.linear_model import ElasticNet, LinearRegression, LassoLars
from sklearn.metrics import r2_score


class Model:
    def fit_and_assess(self, X_train, y_train, X_test, y_test):
        self.fit(X_train, y_train)
        self.assess(X_test, y_test)

        # Flatten the coefficients if necessary
        if self.intercept_ and type(self.intercept_) is not np.float64:
            self.coef_ = self.coef_.ravel()
            self.intercept_ = self.intercept_[0]

    def assess(self, X_test, y_test):
        self.r2_score = self._calculate_r2_score(X_test, y_test)
        self.N_coefs = self._count_nonzero_coefficients()
        self.label = f"Model : $N_\\mathrm{{coeff}} = {self.N_coefs}$, $R^2 = {self.r2_score:.3f}$"

    def _calculate_r2_score(self, X_test, y_test):
        return r2_score(y_test, self.predict(X_test))

    def _count_nonzero_coefficients(self):
        N_intercept = 1
        return np.count_nonzero(self.coef_) + N_intercept

    def refit_model_without_regularization(self, X_train, y_train, X_test, y_test):
        """
        Refits the model using only the nonzero coefficients from the LASSO model,
        without L1-regularization.

        This function takes a fitted LASSO model, identifies the nonzero coefficients,
        and refits a linear model without regularization using only those features.
        The refitted coefficients are then stored back in the original LASSO model object.

        Args:
            X: Input features.
            y: Target values.

        Returns:
            model: The original LASSO model object with coefficients refitted
                   without regularization. The model's coef_ and intercept_
                   attributes are updated.
        """
        # Get the indices of the nonzero coefficients
        nonzero_indices = np.nonzero(self.coef_)[0]

        # If there is only the constant term (inercept_) do not refit
        if len(nonzero_indices) == 0:
            return

        # Select the features corresponding to nonzero coefficients
        X_selected = X_train[:, nonzero_indices]

        # Refitting without regularization (using Ordinary Least Squares)
        refitted_linear_model = LinearRegression(fit_intercept=self.fit_intercept)
        refitted_linear_model.fit(X_selected, y_train)

        # Update the coefficients of the original model
        self.coef_[:] = 0
        self.coef_[nonzero_indices] = refitted_linear_model.coef_

        if self.fit_intercept:
            self.intercept_ = refitted_linear_model.intercept_

        # Re-assess the model
        self.assess(X_test, y_test)


class ElasticNetModel(ElasticNet, Model): ...


class LarsModel(LassoLars, Model): ...
