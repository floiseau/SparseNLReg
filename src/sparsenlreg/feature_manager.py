from typing import List, Union
import itertools

from pandas import DataFrame
import sympy as sp
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sparsenlreg.configuration import LASSOConfig


class FeatureManager:
    def __init__(self, config: LASSOConfig):
        # Input
        self.base_features = config.base_features
        # expansion
        self.expansions = config.expansions
        # Output
        self.output = config.output

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
                # Generate a function from the expression
                expr_sym = sp.sympify(e)
                symbols = list(expr_sym.free_symbols)
                f = sp.lambdify(symbols, expr_sym, "numpy")
                # Evaluate the function
                symbol_names = [symb.name for symb in symbols]
                f_args = (df[name] for name in symbol_names)
                results.append(f(*f_args))
        # Store and return the results
        X = np.array(results) if is_list else np.array(results[0])
        X_names = exprs
        return X, X_names

    def generate_expanded_features(self, df: DataFrame):
        # Generate the base features
        X_base_feat, X_base_feat_names = self._generate_base_features(df)
        # Generate the polynomial features
        X, X_names = self._expand_base_features(X_base_feat, X_base_feat_names, df)
        return X, X_names

    def _generate_base_features(self, df: DataFrame) -> np.ndarray:
        """
        Generate base features from a DataFrame and a list of feature expressions.
        """
        X_base_feat_list, X_base_feat_names = self._evaluate_expression_or_column(
            df, self.base_features
        )
        return np.column_stack(X_base_feat_list), X_base_feat_names

    def _expand_base_features(self, X_base_feat, X_base_feat_names, df):
        """
        Convert an array of base features into an array of transformed features.

        Parameters:
        - X_base_feat: array-like, input features
        - X_base_feat_names: array-like, names of input features
        - pars: dict, dictionnary of the expansion parameters

        Returns:
        - X: ndarray, matrix of polynomial features
        - X_names: list, names of the polynomial features
        """
        # Initialize the expansions
        X = X_base_feat
        X_names = X_base_feat_names

        # Iteratre through the expansions
        for expansion in self.expansions:
            match expansion["type"]:
                case "polynomial":
                    # Extract the parameters
                    degree = expansion["degree"]
                    # Generate polynomial features
                    pf = PolynomialFeatures(degree=degree, include_bias=False)
                    X = pf.fit_transform(X)
                    X_names = pf.get_feature_names_out(X_names)

                case "custom":
                    # Convert the expression into a function
                    expr_sym = sp.sympify(expansion["expression"])
                    symbols = list(expr_sym.free_symbols)
                    f = sp.lambdify(symbols, expr_sym, "numpy")
                    # Extract the variables and parameters
                    vars = {
                        s.name: df[s.name].to_numpy()
                        for s in symbols
                        if s.name in df.columns
                    }
                    # Create all the combinations of parameters
                    par_names = [s.name for s in symbols if s.name in expansion]
                    pars_combinations = itertools.product(
                        *[expansion[n] for n in par_names]
                    )
                    # Create expanded features
                    X_new = []
                    for pars_comb in pars_combinations:
                        # Create the list of arguments
                        par_kwargs = {
                            par_names[i]: pars_comb[i] for i in range(len(par_names))
                        }
                        kwargs = vars | par_kwargs
                        # Evaluate the function to create the new feature
                        X_new.append(f(**kwargs))
                        # Set the feature name
                        name_new = str(expr_sym.subs(par_kwargs))
                        X_names = np.concatenate([X_names, [name_new]])
                    # Update the X array
                    X_new = np.array(X_new).T
                    X = np.concatenate([X, X_new], axis=1)
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
        y, (y_name,) = self._evaluate_expression_or_column(df, self.output)
        return y, y_name
