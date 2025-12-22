from typing import List, Union

from pandas import DataFrame
import sympy as sp
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from sparsenlreg.configuration import LASSOConfig


class FeatureManager:
    def __init__(self, config: LASSOConfig):
        # Input
        self.base_features = config.base_features
        self.base_feature_names = config.base_feature_names
        # Output
        self.output = config.output
        self.output_name = config.output_name
        # Polynomial
        self.degree = config.degree

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

        return results if is_list else results[0]

    def generate_polynomial_features(self, df: DataFrame):
        X_base_feat = self._generate_base_features(df)
        X, X_names = self._transform_base_features_to_polynomial_features(
            X_base_feat, self.degree
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

    def generate_output(self, df: DataFrame) -> np.ndarray:
        """
        Generate output from a DataFrame and an output expression.

        """
        y = self._evaluate_expression_or_column(df, self.output)
        y_name = self.output_name
        return y, y_name
