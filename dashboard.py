import streamlit as st
import pandas as pd
import os
from PIL import Image

st.set_page_config(page_title="Energy Forecasting Model Comparison", layout="wide")
st.title("⚡ Energy Forecasting Model Comparison Dashboard")

# Load metrics
metrics_path = os.path.join('output', 'model_comparison_metrics.csv')
if os.path.exists(metrics_path):
    metrics_df = pd.read_csv(metrics_path)
    st.subheader("Model Performance Metrics")
    st.dataframe(metrics_df.style.format({
        'MSE': '{:.4f}',
        'RMSE': '{:.4f}',
        'MAE': '{:.4f}',
        'R2': '{:.4f}'
    }), use_container_width=True)
else:
    st.warning(f"Metrics file not found at {metrics_path}")

# Show plots for each model
st.subheader("Prediction and Residual Plots")
model_names = ["Conv1D", "LSTM", "GRU"]
for model in model_names:
    st.markdown(f"### {model} Model")
    cols = st.columns(2)
    pred_path = os.path.join('output', f'predictions_{model}.png')
    resid_path = os.path.join('output', f'residuals_{model}.png')
    if os.path.exists(pred_path):
        with cols[0]:
            st.image(pred_path, caption=f"Predictions ({model})", use_container_width=True)
    else:
        with cols[0]:
            st.warning(f"Prediction plot not found for {model}")
    if os.path.exists(resid_path):
        with cols[1]:
            st.image(resid_path, caption=f"Residuals ({model})", use_container_width=True)
    else:
        with cols[1]:
            st.warning(f"Residual plot not found for {model}")

st.info("You can extend this dashboard to include more features, such as uploading new data for prediction or comparing additional models.") 