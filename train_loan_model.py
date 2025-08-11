import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42

class DependentsCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        if "Dependents" in X.columns:
            X["Dependents"] = X["Dependents"].replace("3+", "3")
       
            try:
                X["Dependents"] = pd.to_numeric(X["Dependents"])
            except:
                pass
        return X


def load_data(path="train_u6lujuX_CVtuZ9i.csv"):
    return pd.read_csv(path)


def preprocess_and_train(df):
  
    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])

   
    if "Loan_Status" not in df.columns:
        raise ValueError("Expected 'Loan_Status' column in dataset")
    df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

    X = df.drop(columns=["Loan_Status"])
    y = df["Loan_Status"]

    
    categorical_features = [col for col in X.columns if X[col].dtype == object]
    numeric_features = [col for col in X.columns if col not in categorical_features]

  
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

  
    pipelines = {
        "logreg": Pipeline(steps=[
            ("cleaner", DependentsCleaner()),
            ("preproc", preprocessor),
            ("clf", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))
        ]),
        "rf": Pipeline(steps=[
            ("cleaner", DependentsCleaner()),
            ("preproc", preprocessor),
            ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
        ])
    }

    param_grids = {
        "logreg": {"clf__C": [0.01, 0.1, 1, 10]},
        "rf": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [4, 8, None]
        }
    }

   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    best_models = {}
    for name, pipe in pipelines.items():
        print(f"\nTraining {name} ...")
        grid = GridSearchCV(pipe, param_grids[name], cv=5, scoring="accuracy", n_jobs=-1)
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        y_pred = best.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Best params for {name}: {grid.best_params_}")
        print(f"{name} accuracy on test set: {acc:.4f}")
        print(classification_report(y_test, y_pred))
        best_models[name] = (best, acc)


    best_name, (best_model, best_acc) = max(best_models.items(), key=lambda x: x[1][1])
    print(f"\nSelected best model: {best_name} with accuracy {best_acc:.4f}")

    return best_model, best_name, numeric_features, categorical_features


def save_model(pipeline, filename="loan_model_pipeline.joblib"):
    joblib.dump(pipeline, filename)
    print(f"Saved model pipeline to {filename}")


def main():
    print("Loading data...")
    df = load_data()
    print("Dataset shape:", df.shape)

    best_model, model_name, num_feats, cat_feats = preprocess_and_train(df)
    save_model(best_model)

 
    meta = {
        "model_name": model_name,
        "numeric_features": num_feats,
        "categorical_features": cat_feats
    }
    joblib.dump(meta, "model_metadata.joblib")
    print("Saved model metadata to model_metadata.joblib")


if __name__ == "__main__":
    main()
