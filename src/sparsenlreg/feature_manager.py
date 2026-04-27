from typing import List, Union

from pandas import DataFrame
import sympy as sp
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sparsenlreg.configuration import LASSOConfig


class FeatureManager:
    def __init__(self, config: LASSOConfig):
        # Input
        self.base_features = config.base_features
        self.base_feature_names = config.base_feature_names
        # Output
        self.output = config.output
        self.output_name = config.output_name

    def _evaluate_expression_or_column(
        self,
        df: DataFrame,
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

        return np.array(results) if is_list else np.array(results[0])

    def generate_polynomial_features(self, df: DataFrame, degree: int):
        # Generate the base features
        X_base_feat = self._generate_base_features(df)
        # Generate the polynomial features
        X, X_names = self._transform_base_features_to_polynomial_features(
            X_base_feat, degree
        )
        return X, X_names

    def _generate_base_features(self, df: DataFrame) -> np.ndarray:
        """
        Generate base features from a DataFrame and a list of feature expressions.
        """
        X_base_feat_list = self._evaluate_expression_or_column(df, self.base_features)
        return np.column_stack(X_base_feat_list)

    def _transform_base_features_to_polynomial_features(self, X_base_feat, degree=10):
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
        X_names = pf.get_feature_names_out(self.base_feature_names)
        return X, X_names

    def scale_features(self, X: np.array):
        self.feature_scaler = StandardScaler(with_mean=True)
        return self.feature_scaler.fit_transform(X)

    def scale_output(self, y: np.array):
        self.output_scaler = StandardScaler(with_mean=True)
        y_scaled = self.output_scaler.fit_transform(y.reshape(-1, 1))
        return y_scaled.reshape((-1,))

    def rescale_best_models(self, best_models):
        """Rescale the coefficients of the best models."""
        for i, model in enumerate(best_models):
            model = self.rescale_model(model)
            best_models[i] = model
        return best_models

    def rescale_model(self, model):
        """Rescale the coefficients of a single model."""
        # Extract the coefficients
        coeff = np.array(model.coef_)
        intercept = model.intercept_
        # Get the parameters of the feature scaler
        a = self.feature_scaler.scale_.T
        b = self.feature_scaler.mean_
        # Get the parameters of the output scaler
        A = self.output_scaler.scale_[0]
        B = self.output_scaler.mean_[0]
        # Update the coefficients and intercept
        new_coeff = A * coeff / a
        new_intercept = -A * np.dot(coeff, b / a) + A * intercept + B
        # Update the model
        model.coef_ = new_coeff
        model.intercept_ = new_intercept
        # Return the model
        return model

    def generate_output(self, df: DataFrame) -> np.ndarray:
        """
        Generate output from a DataFrame and an output expression.

        """
        y = self._evaluate_expression_or_column(df, self.output)
        y_name = self.output_name
        return y, y_name
