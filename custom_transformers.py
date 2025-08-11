from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class DependentsCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if 'Dependents' in X.columns:
            X['Dependents'] = X['Dependents'].replace('3+', '3').astype(float)
        return X
