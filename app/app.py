# %%bash
# cat > app.py << 'EOF'

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from huggingface_hub import InferenceClient


# Config & Model Load
st.set_page_config(page_title="Overrun Advisor", layout="wide")

@st.cache_resource
def load_model_and_features():
    model = joblib.load("/content/xgb_tuned_best_param.pkl")
    explainer = shap.TreeExplainer(model)
    feature_cols = list(model.feature_names_in_)
    return model, explainer, feature_cols

model, explainer, FEATURE_COLS = load_model_and_features()


# Split raw vs engineered
RAW_FEATURES = FEATURE_COLS[:16]
ENGINEERED   = FEATURE_COLS[16:]

# Feature descriptions
feature_descriptions = {
    # Raw features...
    'AFP':            'Adjusted Function Points – size of delivered functionality',
    'Input':          'Number of input screens or transactions',
    'Output':         'Number of output reports or transactions',
    'Enquiry':        'Number of enquiry transactions (read-only screens)',
    'File':           'Number of logical files (data stores) used',
    'Interface':      'Number of external system interfaces',
    'Added':          'Count of function points added after initial scope',
    'Changed':        'Count of function points modified after initial scope',
    'Deleted':        'Count of function points removed after initial scope',
    'PDR_AFP':        'AFP at Preliminary Design Review – size agreed at PDR',
    'PDR_UFP':        'Unadjusted Function Points at PDR',
    'NPDR_AFP':       'AFP at “no‐PDR” or first build – size after major changes',
    'NPDU_UFP':       'Unadjusted Function Points at NPDR stage',
    'Resource':       'Number of team members assigned',
    'Dev.Type':       'Development type flag (e.g. 0=enhancement, 1=new development)',
    'Duration':       'Schedule length in months',
    'N_effort':       'Nominal effort estimate (person‐months) before overrun',
    'Effort':         'Actual effort expended (person‐months)',

    # Engineered features
    'overrun_pct':    'Percent effort overrun: (Effort/N_effort) – 1',
    'churn_rate':     'Scope churn as fraction of size: (Added+Changed+Deleted)/AFP',
    'pdr_ratio':      'Proportion signed‐off at PDR: PDR_AFP/AFP',
    'prod_nominal':   'Nominal productivity: AFP per planned person‐month',
    'prod_actual':    'Actual productivity: AFP per actual person‐month',
    'team_intensity': 'Team size per month: Resource/Duration',
    'churn_x_team':   'Interaction of churn and team: churn_rate × team_intensity',
}

# LLM client
HF_API_KEY = os.getenv("HF_API_KEY")
client = InferenceClient(provider="hyperbolic", api_key=HF_API_KEY)

def call_llm(prompt,
             model_name="deepseek-ai/DeepSeek-R1",
             temperature=0.5, max_tokens=200, top_p=0.7):
    # Add a system message to forbid chain-of-thought and enforce concise bullets
    messages = [
        {
            "role": "system",
            "content": (
                "You are an advisor. Produce exactly one bullet per feature, "
                "one sentence each, in the same order as given. "
                "Do NOT show your chain of thought or any analysis."
            )
        },
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stream=False
    )
    return response.choices[0].message.content.strip()

def build_llm_prompt(top5_shap):
    prompt_lines = ["Here are the top 5 factors driving project effort overruns, listed with their numeric SHAP impacts:"]
    for feat, val in top5_shap.iloc[:5].items():
        desc = feature_descriptions.get(feat, "")
        prompt_lines.append(f"- **{feat}** ({desc}) with impact {val:.4f}")
    prompt_lines.append(
        "\nFor each factor above, provide one concise, actionable tip to help reduce effort overruns. "
        "Use the numeric impact values to prioritize and—if helpful—mention the magnitude in your tip. "
        "Return exactly five bullets, one sentence each, in the same order."

    )
    return "\n".join(prompt_lines)

# UI
st.sidebar.header("Upload Project CSV")
uploaded = st.sidebar.file_uploader("CSV with raw features only", type="csv")

st.title("📊 Project Overrun Advisor")

if uploaded:
    df = pd.read_csv(uploaded)

    # ensure raw inputs
    missing = set(RAW_FEATURES) - set(df.columns)
    if missing:
        st.error(f"Missing columns: {sorted(missing)}")
        st.stop()

    # compute engineered
    df['churn_rate']     = (df['Added'] + df['Changed'] + df['Deleted']) / df['AFP']
    df['pdr_ratio']      = df['PDR_AFP'] / df['AFP']
    df['prod_nominal']   = df['AFP'] / df['N_effort']
    df['team_intensity'] = df['Resource'] / df['Duration']

    # build X and predict
    X = df[FEATURE_COLS]
    df['predicted_overrun_pct'] = model.predict(X)

    st.subheader("Predictions")
    st.dataframe(df[['predicted_overrun_pct'] + RAW_FEATURES])

    # SHAP analysis
    shap_vals = explainer.shap_values(X)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    shap_ser = pd.Series(mean_abs, index=FEATURE_COLS).sort_values(ascending=False).iloc[:5]

    st.subheader("Global SHAP Feature Importance")
    plt.figure(figsize=(8, 4))
    shap.summary_plot(
        shap_vals,
        X,
        plot_type="bar",
        max_display=10,
        show=False
    )
    st.pyplot(plt.gcf())

    st.subheader("SHAP Beeswarm Plot")
    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_vals,
        X,
        max_display=10,
        show=False
    )
    st.pyplot(plt.gcf())

    # LLM tips
    st.subheader("💡 AI Recommendations")
    if st.button("Get AI Tips"):
        prompt = build_llm_prompt(shap_ser)
        with st.spinner("Generating tips…"):
            tips = call_llm(prompt)
        st.markdown(tips)
