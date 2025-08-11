# predict_cli.py
import joblib
import pandas as pd
import argparse

def load_model(path="loan_model_pipeline.joblib"):
    return joblib.load(path)

def main(args):
    model = load_model(args.model)
  
    data = {k: [v] for k, v in vars(args).items() if k != 'model'}
    df = pd.DataFrame(data)
    pred = model.predict(df)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[:,1][0]
    print("Prediction:", "Approved (Y)" if pred==1 else "Rejected (N)")
    if proba is not None:
        print(f"Approval prob: {proba:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="loan_model_pipeline.joblib")
  
    parser.add_argument("--ApplicantIncome", type=float, default=2500.0)
    parser.add_argument("--CoapplicantIncome", type=float, default=0.0)
    parser.add_argument("--LoanAmount", type=float, default=120.0)
    parser.add_argument("--Loan_Amount_Term", type=float, default=360.0)
    parser.add_argument("--Credit_History", type=float, default=1.0)
    parser.add_argument("--Gender", type=str, default="Male")
    parser.add_argument("--Married", type=str, default="No")
    parser.add_argument("--Dependents", type=str, default="0")
    parser.add_argument("--Education", type=str, default="Graduate")
    parser.add_argument("--Self_Employed", type=str, default="No")
    parser.add_argument("--Property_Area", type=str, default="Urban")
    args = parser.parse_args()
    main(args)
