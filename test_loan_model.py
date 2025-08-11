import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin


class DependentsCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        if 'Dependents' in X.columns:
            X['Dependents'] = X['Dependents'].replace('3+', '3')
            try:
                X['Dependents'] = pd.to_numeric(X['Dependents'])
            except Exception:
                pass
        return X


df_test = pd.read_csv("test_Y3wMUE5_7gLdaTN.csv")


if "Loan_ID" in df_test.columns:
    df_test = df_test.drop(columns=["Loan_ID"])

y_true = None
if "Loan_Status" in df_test.columns:
    y_true = df_test["Loan_Status"].map({"Y": 1, "N": 0})
    X_test = df_test.drop(columns=["Loan_Status"])
else:
    X_test = df_test


model = joblib.load("loan_model_pipeline.joblib")


y_pred = model.predict(X_test)


if y_true is not None:
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))
else:
    print("Predictions:", y_pred)
