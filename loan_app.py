import streamlit as st
import pandas as pd
import joblib
import numpy as np
from custom_transformers import DependentsCleaner

@st.cache_resource
def load_artifacts(model_path="loan_model_pipeline.joblib", meta_path="model_metadata.joblib"):
    model = joblib.load(model_path)
    meta = joblib.load(meta_path)
    return model, meta

def input_form(meta):
    st.sidebar.header("Applicant Input")

    
    num_feats = meta.get('numeric_features', ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History'])
    cat_feats = meta.get('categorical_features', ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'])

    data = {}
    
    for col in num_feats:
       
        if col.lower().find('income') >= 0:
            val = st.sidebar.number_input(col, min_value=0.0, max_value=1e6, value=2500.0, step=100.0)
        elif col.lower().find('loan') >= 0:
            val = st.sidebar.number_input(col, min_value=0.0, max_value=1e6, value=120.0, step=1.0)
        elif col.lower().find('credit') >= 0:
            val = st.sidebar.selectbox(col, options=[0.0, 1.0], index=1)
        else:
            val = st.sidebar.number_input(col, min_value=0.0, max_value=1e6, value=0.0, step=1.0)
        data[col] = [val]

   
    for col in cat_feats:
        if col.lower() == 'gender':
            data[col] = [st.sidebar.selectbox("Gender", ["Male", "Female"], index=0)]
        elif col.lower() == 'married':
            data[col] = [st.sidebar.selectbox("Married", ["Yes", "No"], index=1)]
        elif col.lower() == 'education':
            data[col] = [st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"], index=0)]
        elif col.lower() == 'self_employed':
            data[col] = [st.sidebar.selectbox("Self Employed", ["Yes", "No"], index=1)]
        elif col.lower() == 'property_area':
            data[col] = [st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"], index=0)]
        elif col.lower() == 'dependents':
            data[col] = [st.sidebar.selectbox("Dependents", ["0","1","2","3+"], index=0)]
        else:
           
            data[col] = [st.sidebar.text_input(col, value="")]

    return pd.DataFrame.from_dict(data)

def main():
    st.title("Loan Approval Prediction")
    st.write("Enter applicant details on the left and click Predict.")

    model, meta = load_artifacts()

    input_df = input_form(meta)
    st.subheader("Input sample")
    st.write(input_df.T)

    if st.button("Predict"):
        preds = model.predict(input_df)
        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[:, 1]  

        label = "Approved (Y)" if preds[0] == 1 else "Rejected (N)"
        st.markdown(f"## Prediction: {label}")
        if probs is not None:
            st.info(f"Approval probability: {probs[0]*100:.1f}%")

    
        st.write("Model:", meta.get("model_name", "pipeline"))

if __name__ == "__main__":
    main()


"""
 uv run streamlit run loan_app.py
"""