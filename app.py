
import streamlit as st
import pandas as pd
import shap
import joblib
from xgboost import XGBClassifier

# 標題與說明
st.set_page_config(page_title="Psychiatric Dropout Risk", layout="wide")
st.title("🧠 Psychiatric Dropout Risk Predictor")
st.markdown("Estimate 3-month dropout risk after psychiatric discharge with SHAP explanations.")

# 載入模型
@st.cache_resource
def load_model():
    return joblib.load("dropout_model.pkl")

model = load_model()

# 載入 SHAP explainer
@st.cache_resource
def load_explainer():
    X_sample = pd.read_csv("sample_input.csv")
    return shap.Explainer(model), X_sample.columns.tolist()

explainer, feature_names = load_explainer()

# 使用者輸入欄位
st.sidebar.header("Input Patient Information")
def user_input():
    data = {
        'age': st.sidebar.slider("Age", 18, 75, 30),
        'gender_Female': st.sidebar.selectbox("Gender", ["Male", "Female"]) == "Female",
        'diagnosis_Bipolar': st.sidebar.checkbox("Diagnosis: Bipolar"),
        'diagnosis_Depression': st.sidebar.checkbox("Diagnosis: Depression"),
        'diagnosis_Schizophrenia': True,  # Default true if others not selected
        'length_of_stay': st.sidebar.slider("Length of stay (days)", 1, 90, 14),
        'num_previous_admissions': st.sidebar.slider("Previous admissions", 0, 10, 2),
        'has_social_worker_No': st.sidebar.radio("Social worker assigned?", ["Yes", "No"]) == "No",
        'medication_compliance_score': st.sidebar.slider("Medication compliance (0-10)", 0.0, 10.0, 5.0),
        'has_recent_self_harm_Yes': st.sidebar.checkbox("Recent self-harm history"),
        'self_harm_during_admission_Yes': st.sidebar.checkbox("Self-harm during admission"),
        'family_support_score': st.sidebar.slider("Family support (0–10)", 0, 10, 5),
        'post_discharge_followups': st.sidebar.slider("Follow-ups in 30 days", 0, 5, 1)
    }

    df = pd.DataFrame([data])
    # One-hot adjustments
    df['gender_Male'] = not data['gender_Female']
    df['diagnosis_Schizophrenia'] = not (data['diagnosis_Bipolar'] or data['diagnosis_Depression'])
    df['has_social_worker_Yes'] = not data['has_social_worker_No']
    df['has_recent_self_harm_No'] = not data['has_recent_self_harm_Yes']
    df['self_harm_during_admission_No'] = not data['self_harm_during_admission_Yes']
    return df

input_df = user_input()
X_final = input_df.reindex(columns=feature_names, fill_value=0)

# 預測與風險顯示
prob = model.predict_proba(X_final)[0][1]
score = int(round(prob * 100))
st.subheader("Predicted Dropout Risk Score")
st.metric(label="Dropout risk (3 months)", value=f"{score}/100")

# SHAP 圖
st.subheader("SHAP Explanation")
shap_values = explainer(X_final)
st.set_option('deprecation.showPyplotGlobalUse', False)
shap.plots.waterfall(shap_values[0], max_display=8)
st.pyplot(bbox_inches='tight')
