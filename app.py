# ✅ psychiatric_dropout demo App（穩定版：修復 SHAP 畫圖錯誤）
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import xgboost as xgb

st.set_page_config(page_title="Psychiatric Dropout Risk", layout="wide")
st.title("🧠 Psychiatric Dropout Risk Predictor")

# 載入模型與欄位樣板
model = joblib.load("dropout_model.pkl")
sample = pd.read_csv("sample_input.csv")

# 使用者輸入表單
with st.sidebar:
    st.header("Patient Info")
    age = st.slider("Age", 18, 75, 35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    diagnosis = st.selectbox("Diagnosis", [
        "Schizophrenia", "Bipolar", "Depression",
        "Personality Disorder", "Substance Use Disorder", "Dementia"
    ])
    length_of_stay = st.slider("Length of Stay (days)", 1, 90, 10)
    num_adm = st.slider("# Previous Admissions", 0, 15, 1)
    social_worker = st.radio("Has Social Worker", ["Yes", "No"])
    compliance = st.slider("Medication Compliance Score", 0.0, 10.0, 5.0)
    recent_self_harm = st.radio("Recent Self-harm", ["Yes", "No"])
    selfharm_adm = st.radio("Self-harm During Admission", ["Yes", "No"])
    support = st.slider("Family Support Score", 0.0, 10.0, 5.0)
    followups = st.slider("Post-discharge Followups", 0, 10, 2)

# 建立單筆使用者資料
user_input = pd.DataFrame({
    'age': [age],
    'length_of_stay': [length_of_stay],
    'num_previous_admissions': [num_adm],
    'medication_compliance_score': [compliance],
    'family_support_score': [support],
    'post_discharge_followups': [followups],
    f'gender_{gender}': [1],
    f'diagnosis_{diagnosis}': [1],
    f'has_social_worker_{social_worker}': [1],
    f'has_recent_self_harm_{recent_self_harm}': [1],
    f'self_harm_during_admission_{selfharm_adm}': [1],
})

# 對齊 sample 欄位
X_final = pd.DataFrame(columns=sample.columns)
X_final.loc[0] = 0  # 全欄位預設為 0
for col in user_input.columns:
    if col in X_final.columns:
        X_final.at[0, col] = user_input[col][0]

# 使用 numpy 並避免特徵驗證錯誤（透過 validate_features=False）
prob = model.predict_proba(X_final, validate_features=False)[0][1]
st.metric("Predicted Dropout Risk (within 3 months)", f"{prob*100:.1f}%")

# 分級提示
if prob > 0.7:
    st.error("🔴 High Risk")
elif prob > 0.4:
    st.warning("🟡 Medium Risk")
else:
    st.success("🟢 Low Risk")

# SHAP 解釋圖（避免畫圖錯誤）
st.subheader("SHAP Explanation")
explainer = shap.Explainer(model)
shap_values = explainer(X_final)
fig = plt.figure()
shap.summary_plot(shap_values, X_final, show=False)
st.pyplot(fig)

st.caption("Model trained on simulated data reflecting clinical dropout risk factors. Not for clinical use.")
