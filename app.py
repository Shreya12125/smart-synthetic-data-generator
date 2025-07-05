import streamlit as st
import pandas as pd
from sdv.tabular import GaussianCopula
import matplotlib.pyplot as plt
import io
import re

# ---------- Page config ----------
st.set_page_config(page_title="Smart Synthetic Data Generator",
                   page_icon="🧠",
                   layout="wide")

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV", type=["csv"])
num_rows = st.sidebar.number_input("📌 Rows to generate", 10, 10_000, 100, step=10)
generate_btn = st.sidebar.button("🚀 Generate Synthetic Data", use_container_width=True)

with st.sidebar.expander("ℹ️ About this app"):
    st.markdown(
        "Generate **privacy‑friendly synthetic data** from any tabular dataset. "
        "Built with Streamlit + SDV (GaussianCopula)."
    )

# ---------- Helpers ----------
def detect_pii_column(col, series):
    name_tokens = [
        "name", "email", "phone", "mobile", "contact",
        "ssn", "dob", "address", "card"
    ]
    if any(tok in col.lower() for tok in name_tokens):
        return True
    sample = series.dropna().astype(str).head(10).tolist()
    patterns = [
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        r"\d{3}[-.\s]?\d{2}[-.\s]?\d{4}",            # SSN
        r"\b\d{10,16}\b",                            # phone/card
    ]
    return any(re.search(pat, x) for x in sample for pat in patterns)

# ---------- Main ----------
st.title("🧠 Smart Synthetic Data Generator")

if uploaded_file:
    original_df = pd.read_csv(uploaded_file)

    # --- Train & sample if user clicked ---
    if generate_btn:
        with st.spinner("Training GaussianCopula…"):
            model = GaussianCopula()
            model.fit(original_df)
            synthetic_df = model.sample(num_rows)
        st.success("Synthetic data generated!")

    # --- Build UI tabs ---
    tabs = st.tabs(["📋 Overview", "📑 Original Data", "🧪 Synthetic Data", "📊 Charts"])

    # --- Overview tab ---
    with tabs[0]:
        st.header("Dataset Overview")

        with st.expander("🔎 Quick Stats", expanded=True):
            st.write(f"**Rows:** {original_df.shape[0]} &nbsp;&nbsp;|&nbsp;&nbsp; **Columns:** {original_df.shape[1]}")
            st.metric("Missing cells", int(original_df.isna().sum().sum()))
            st.dataframe(original_df.describe(include="all").T, use_container_width=True)

        with st.expander("🔐 PII Scan", expanded=False):
            pii_cols = [c for c in original_df.columns if detect_pii_column(c, original_df[c])]
            if pii_cols:
                st.warning("⚠️ Potential PII columns: " + ", ".join(pii_cols))
            else:
                st.success("✅ No obvious PII columns detected.")

    # --- Original Data tab ---
    with tabs[1]:
        st.header("Original Data Preview")
        st.dataframe(original_df, use_container_width=True, height=400)

    # --- Synthetic Data tab (only if generated) ---
    with tabs[2]:
        st.header("Synthetic Data Preview")
        if generate_btn:
            st.dataframe(synthetic_df, use_container_width=True, height=400)
            buf = io.StringIO()
            synthetic_df.to_csv(buf, index=False)
            st.download_button("📥 Download CSV", buf.getvalue(),
                               "synthetic_data.csv", "text/csv", use_container_width=True)
        else:
            st.info("Generate data from the sidebar to see results here.")

    # --- Charts tab ---
    with tabs[3]:
        st.header("Distribution Comparison")
        if generate_btn:
            num_cols = original_df.select_dtypes("number").columns
            if num_cols.empty:
                st.info("No numeric columns to chart.")
            else:
                for col in num_cols:
                    fig, ax = plt.subplots()
                    original_df[col].plot(kind="kde", ax=ax, label="Original")
                    synthetic_df[col].plot(kind="kde", ax=ax, label="Synthetic")
                    ax.set_title(f"Distribution • {col}")
                    ax.legend()
                    st.pyplot(fig, use_container_width=True)
        else:
            st.info("Generate data from the sidebar to view charts.")

else:
    st.info("👈 Upload a CSV file in the sidebar to begin.")
