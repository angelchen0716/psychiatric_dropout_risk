# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
import json
from io import BytesIO

st.set_page_config(page_title="Psychiatric Dropout Risk", layout="wide")
st.title("🧠 Psychiatric Dropout Risk Predictor")

# -----------------------------
# Load model & sample schema
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("dropout_model.pkl")

@st.cache_data
def load_sample():
    return pd.read_csv("sample_input.csv")

model = load_model()
sample = load_sample()
feature_cols = list(sample.columns)

# -----------------------------
# Optional: load thresholds.json
# -----------------------------
def load_thresholds_from_file():
    try:
        with open("thresholds.json", "r", encoding="utf-8") as f:
            js = json.load(f)
        # expect keys: low_medium, medium_high
        return float(js.get("low_medium", 0.4)), float(js.get("medium_high", 0.7))
    except Exception:
        return 0.4, 0.7  # defaults

default_low_med, default_med_high = load_thresholds_from_file()

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Settings")
    st.caption("Thresholds & clinical override")

    low_med = st.slider("Probability threshold: Low → Medium", 0.05, 0.95, float(default_low_med), 0.01)
    med_high = st.slider("Probability threshold: Medium → High", 0.05, 0.95, float(default_med_high), 0.01)
    if med_high <= low_med:
        st.error("⚠️ Medium→High 閾值需大於 Low→Medium 閾值。已自動調整。")
        med_high = max(low_med + 0.05, 0.05)

    use_clinical_override = st.checkbox("Enable clinical override for self-harm", value=True)
    min_tier_on_selfharm = st.selectbox(
        "Minimum risk tier when self-harm is present",
        ["Medium", "High"], index=0
    )

    st.divider()
    st.caption("Input form defaults")
    default_age = st.slider("Default Age", 18, 75, 35)
    default_los = st.slider("Default Length of Stay (days)", 1, 90, 10)

st.info("Note: Model trained on simulated data reflecting clinical dropout risk factors. Not for clinical use.")

# -----------------------------
# Utilities
# -----------------------------
TIER_ORDER = ["Low", "Medium", "High"]

def apply_tier(prob, low_med, med_high):
    if prob > med_high:
        return "High"
    elif prob > low_med:
        return "Medium"
    else:
        return "Low"

def escalate_by_rule(base_tier, has_recent_sh, has_adm_sh, enabled=True, min_tier="Medium"):
    if not enabled:
        return base_tier
    if bool(has_recent_sh) or bool(has_adm_sh):
        # enforce minimum tier
        if TIER_ORDER.index(base_tier) < TIER_ORDER.index(min_tier):
            return min_tier
    return base_tier

def align_to_schema(df_input: pd.DataFrame, schema_cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(columns=schema_cols)
    out.loc[0 if df_input.shape[0]==1 else range(df_input.shape[0]), :] = 0
    for col in df_input.columns:
        if col in out.columns:
            out.loc[:len(df_input)-1, col] = df_input[col].values
    # ensure numeric types for model
    out = out.apply(pd.to_numeric, errors="coerce").fillna(0)
    return out

def predict_proba_on_df(X_df: pd.DataFrame) -> np.ndarray:
    # xgboost sklearn API supports validate_features=False to allow column mismatch order
    return model.predict_proba(X_df, validate_features=False)[:, 1]

# -----------------------------
# SHAP setup (robust)
# -----------------------------
@st.cache_resource
def make_shap_explainer(m):
    try:
        # Try native tree explainer first
        return shap.TreeExplainer(m)
    except Exception:
        # Some saved models need the booster
        try:
            booster = m.get_booster()
            return shap.TreeExplainer(booster)
        except Exception:
            # Fallback to general Explainer (slower)
            return shap.Explainer(m)

explainer = make_shap_explainer(model)

def plot_shap_summary(X_for_summary):
    fig = plt.figure()
    try:
        shap_values = explainer(X_for_summary)
        shap.summary_plot(shap_values, X_for_summary, show=False)
    except Exception:
        # Legacy API fallback
        sv = explainer.shap_values(X_for_summary)
        shap.summary_plot(sv, X_for_summary, show=False)
    st.pyplot(fig, clear_figure=True)

def plot_shap_waterfall(X_one_row):
    st.caption("Per-patient explanation (waterfall)")
    try:
        sv = explainer(X_one_row)
        fig = shap.plots._waterfall.waterfall_legacy(
            shap_values=sv.values[0],
            feature_names=X_one_row.columns.tolist(),
            max_display=12,
            show=False
        )
        st.pyplot(fig, clear_figure=True)
    except Exception:
        # Legacy path
        sv = explainer.shap_values(X_one_row)
        base = getattr(explainer, "expected_value", 0.0)
        # Build waterfall manually
        vals = sv[0] if isinstance(sv, list) else sv[0]
        contrib = pd.Series(vals, index=X_one_row.columns).sort_values(key=np.abs, ascending=False)[:12]
        base_prob = 1/(1+np.exp(-base)) if isinstance(base, (int,float)) else 0.5
        st.write("Top contributors:", contrib.to_frame("SHAP value").head(12))

# -----------------------------
# Tabs: Single / Batch
# -----------------------------
tab_single, tab_batch, tab_global = st.tabs(["🧍 Single Patient", "📥 Batch Upload", "📊 Global SHAP"])

with tab_single:
    # ----- User input form -----
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 18, 75, int(default_age))
        gender = st.selectbox("Gender", ["Male", "Female"])
        diagnosis = st.selectbox("Diagnosis", [
            "Schizophrenia", "Bipolar", "Depression",
            "Personality Disorder", "Substance Use Disorder", "Dementia"
        ])
    with col2:
        length_of_stay = st.slider("Length of Stay (days)", 1, 90, int(default_los))
        num_adm = st.slider("# Previous Admissions", 0, 15, 1)
        social_worker = st.radio("Has Social Worker", ["Yes", "No"], horizontal=True)
    with col3:
        compliance = st.slider("Medication Compliance Score", 0.0, 10.0, 5.0, 0.1)
        support = st.slider("Family Support Score", 0.0, 10.0, 5.0, 0.1)
        followups = st.slider("Post-discharge Followups", 0, 10, 2)
        recent_self_harm = st.radio("Recent Self-harm", ["Yes", "No"], horizontal=True)
        selfharm_adm = st.radio("Self-harm During Admission", ["Yes", "No"], horizontal=True)

    # One-hot pack for single row
    user_input = {
        'age': age,
        'length_of_stay': length_of_stay,
        'num_previous_admissions': num_adm,
        'medication_compliance_score': compliance,
        'family_support_score': support,
        'post_discharge_followups': followups,
        f'gender_{gender}': 1,
        f'diagnosis_{diagnosis}': 1,
        f'has_social_worker_{social_worker}': 1,
        f'has_recent_self_harm_{recent_self_harm}': 1,
        f'self_harm_during_admission_{selfharm_adm}': 1,
    }
    user_df = pd.DataFrame([user_input])

    X_final = align_to_schema(user_df, feature_cols)
    prob = float(predict_proba_on_df(X_final)[0])
    base_tier = apply_tier(prob, low_med, med_high)

    has_recent = int(user_df.get("has_recent_self_harm_Yes", 0))
    has_adm = int(user_df.get("self_harm_during_admission_Yes", 0))
    final_tier = escalate_by_rule(
        base_tier,
        has_recent,
        has_adm,
        enabled=use_clinical_override,
        min_tier=min_tier_on_selfharm
    )

    st.metric("Predicted Dropout Risk (within 3 months)", f"{prob*100:.1f}%")

    if final_tier == "High":
        st.error("🔴 High Risk")
    elif final_tier == "Medium":
        st.warning("🟡 Medium Risk")
    else:
        st.success("🟢 Low Risk")

    # SHAP (per-patient)
    with st.expander("🔍 SHAP Explanation (this patient)"):
        plot_shap_waterfall(X_final)

with tab_batch:
    st.write("Upload a CSV or Excel with patient rows. Columns will be aligned to `sample_input.csv` schema.")
    up = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if up is not None:
        if up.name.endswith(".xlsx"):
            df_raw = pd.read_excel(up)
        else:
            df_raw = pd.read_csv(up)

        # Allow simple wide-to-expected mapping: if user provides raw fields (age, gender, etc.)
        # we try to one-hot minimal fields into the expected schema
        df_proc = df_raw.copy()

        # If the uploaded file looks like "raw" fields, expand them to one-hot
        # (safe no-op if columns already one-hot)
        def maybe_one_hot(df):
            df = df.copy()
            # Gender
            if "gender" in df.columns:
                for g in ["Male", "Female"]:
                    df[f"gender_{g}"] = (df["gender"].astype(str) == g).astype(int)
            # Diagnosis
            if "diagnosis" in df.columns:
                diags = ["Schizophrenia","Bipolar","Depression","Personality Disorder","Substance Use Disorder","Dementia"]
                for d in diags:
                    df[f"diagnosis_{d}"] = (df["diagnosis"].astype(str) == d).astype(int)
            # Social worker
            if "has_social_worker" in df.columns:
                for y in ["Yes","No"]:
                    df[f"has_social_worker_{y}"] = (df["has_social_worker"].astype(str) == y).astype(int)
            # Self-harm flags
            if "has_recent_self_harm" in df.columns:
                for y in ["Yes","No"]:
                    df[f"has_recent_self_harm_{y}"] = (df["has_recent_self_harm"].astype(str) == y).astype(int)
            if "self_harm_during_admission" in df.columns:
                for y in ["Yes","No"]:
                    df[f"self_harm_during_admission_{y}"] = (df["self_harm_during_admission"].astype(str) == y).astype(int)
            return df

        df_proc = maybe_one_hot(df_proc)

        X_batch = align_to_schema(df_proc, feature_cols)
        probs = predict_proba_on_df(X_batch)

        # derive tiers and override
        has_recent_arr = (X_batch.get("has_recent_self_harm_Yes", pd.Series([0]*len(X_batch))).astype(int)).values
        has_adm_arr = (X_batch.get("self_harm_during_admission_Yes", pd.Series([0]*len(X_batch))).astype(int)).values

        base_tiers = [apply_tier(p, low_med, med_high) for p in probs]
        final_tiers = [
            escalate_by_rule(bt, has_recent_arr[i], has_adm_arr[i], enabled=use_clinical_override, min_tier=min_tier_on_selfharm)
            for i, bt in enumerate(base_tiers)
        ]

        out = df_raw.copy()
        out["dropout_prob"] = np.round(probs, 4)
        out["risk_tier_base"] = base_tiers
        out["risk_tier_final"] = final_tiers

        st.success(f"Processed {len(out)} rows.")
        st.dataframe(out, use_container_width=True)

        # Download button
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download predictions (CSV)",
            data=csv_bytes,
            file_name="predictions_dropout.csv",
            mime="text/csv"
        )

with tab_global:
    st.write("Global feature importance via SHAP summary (using sample schema as background).")
    # Use a small background for speed if sample is large
    bg = sample.copy()
    if len(bg) > 500:
        bg = bg.sample(500, random_state=42)
    plot_shap_summary(bg)
# =============================
# 📈 Evaluation & Model Card
# =============================
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, precision_recall_curve,
    classification_report, brier_score_loss, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.calibration import calibration_curve

tab_eval, = st.tabs(["📈 Evaluation & Model Card"])

def plot_roc(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    st.pyplot(fig, clear_figure=True)
    return roc_auc

def plot_pr(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig = plt.figure()
    plt.step(recall, precision, where="post")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    st.pyplot(fig, clear_figure=True)

def plot_confmat(y_true, y_score, thr):
    y_pred = (y_score >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    fig = plt.figure()
    im = plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix @ threshold={thr:.2f}")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = [0,1]
    plt.xticks(tick_marks, ["No dropout", "Dropout"])
    plt.yticks(tick_marks, ["No dropout", "Dropout"])
    # annotate counts
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    st.pyplot(fig, clear_figure=True)
    # metrics
    sens = tp / (tp + fn) if (tp+fn)>0 else 0.0
    spec = tn / (tn + fp) if (tn+fp)>0 else 0.0
    ppv  = tp / (tp + fp) if (tp+fp)>0 else 0.0
    npv  = tn / (tn + fn) if (tn+fn)>0 else 0.0
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Sensitivity (Recall)": sens,
        "Specificity": spec,
        "PPV (Precision)": ppv,
        "NPV": npv
    }

def plot_calibration(y_true, y_score, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="quantile")
    fig = plt.figure()
    plt.plot([0,1],[0,1], linestyle="--")
    plt.plot(prob_pred, prob_true, marker="o")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration Plot")
    st.pyplot(fig, clear_figure=True)
    return prob_true, prob_pred

with tab_eval:
    st.subheader("Upload labeled evaluation set")
    st.write("上傳含 **真實標籤** 的 CSV 或 Excel。請提供目標欄位，例如：`dropout_within_3mo`（0/1）。")
    file_eval = st.file_uploader("Upload CSV or Excel with labels", type=["csv","xlsx"], key="eval_upl")
    label_col = st.text_input("Name of label column (0/1)", "dropout_within_3mo")
    thr_for_confmat = st.slider("Threshold for confusion matrix", 0.05, 0.95, float((low_med+med_high)/2), 0.01)

    if file_eval is not None:
        df_raw = pd.read_excel(file_eval) if file_eval.name.endswith(".xlsx") else pd.read_csv(file_eval)
        if label_col not in df_raw.columns:
            st.error(f"Label column `{label_col}` not found.")
        else:
            # keep label and make copy for features
            y_true = df_raw[label_col].astype(int).values
            df_feat = df_raw.drop(columns=[label_col]).copy()

            # same one-hot helper as batch tab
            df_feat = (lambda df: (
                (lambda d: (
                    d.assign(**{f"gender_{g}": (d["gender"].astype(str)==g).astype(int) for g in ["Male","Female"]}) if "gender" in d.columns else d
                ))(
                    (lambda d: (
                        d.assign(**{f"diagnosis_{di}": (d["diagnosis"].astype(str)==di).astype(int)
                                    for di in ["Schizophrenia","Bipolar","Depression","Personality Disorder","Substance Use Disorder","Dementia"]})
                        if "diagnosis" in d.columns else d
                    ))(
                        (lambda d: (
                            d.assign(**{f"has_social_worker_{y}": (d["has_social_worker"].astype(str)==y).astype(int) for y in ["Yes","No"]})
                            if "has_social_worker" in d.columns else d
                        ))(
                            (lambda d: (
                                d.assign(**{f"has_recent_self_harm_{y}": (d["has_recent_self_harm"].astype(str)==y).astype(int) for y in ["Yes","No"]})
                                if "has_recent_self_harm" in d.columns else d
                            ))(
                                (lambda d: (
                                    d.assign(**{f"self_harm_during_admission_{y}": (d["self_harm_during_admission"].astype(str)==y).astype(int) for y in ["Yes","No"]})
                                    if "self_harm_during_admission" in d.columns else d
                                ))(df_feat)
                            )
                        )
                    )
                )
            ))(df_feat)

            X_eval = align_to_schema(df_feat, feature_cols)
            y_score = predict_proba_on_df(X_eval)

            st.markdown("### Metrics & Plots")
            roc_auc = plot_roc(y_true, y_score)
            plot_pr(y_true, y_score)
            cm_stats = plot_confmat(y_true, y_score, thr_for_confmat)
            prob_true, prob_pred = plot_calibration(y_true, y_score)
            brier = brier_score_loss(y_true, y_score)

            # summary table
            st.markdown("#### Summary")
            st.write({
                "AUC": round(roc_auc, 3),
                "Brier score (lower better)": round(brier, 4),
                **{k: (round(v,3) if isinstance(v,float) else v) for k,v in cm_stats.items()}
            })

            # downloadable CSV of per-row predictions
            out_eval = df_raw.copy()
            out_eval["pred_prob"] = y_score
            out_eval["pred_label_thr"] = (y_score >= thr_for_confmat).astype(int)
            st.download_button(
                "⬇️ Download per-row predictions (CSV)",
                data=out_eval.to_csv(index=False).encode("utf-8"),
                file_name="eval_predictions.csv",
                mime="text/csv"
            )

            st.divider()
            st.markdown("### Model Card (English)")
            st.markdown(f"""
**Model**: XGBoost binary classifier  
**Target**: 3‑month psychiatric outpatient dropout (1=yes, 0=no)  
**Inputs**: Demographics, diagnosis group, length of stay, prior admissions, self-harm flags, social worker, family support, medication compliance, post‑discharge follow-ups (one‑hot as in `sample_input.csv`).  
**Training data**: Simulated clinical-like dataset (non-identifiable).  
**Intended use**: Educational demo for risk stratification UI and explainability. **Not for clinical use**.

**Performance (on uploaded test set)**  
- ROC AUC: {roc_auc:.3f}  
- Brier score: {brier:.4f} (lower is better)  
- Confusion matrix @ threshold {thr_for_confmat:.2f}: {cm_stats}  

**Calibration**: See calibration plot; consider Platt scaling or isotonic regression if miscalibrated.  

**Risk tiers**: Default Low/Medium/High cutoffs = {low_med:.2f}/{med_high:.2f}. A clinical override can enforce a minimum tier when recent or in‑admission self‑harm is present (configurable).  

**Fairness & safety**: This demo may reflect dataset biases and should not be used to guide real care decisions. Always pair with clinician judgment and post‑discharge care pathways.  
            """)

st.caption("Figures: ROC, PR, Confusion Matrix, and Calibration curves are generated on the uploaded labeled set. SHAP: waterfall (per‑patient) and beeswarm summary (global).")
