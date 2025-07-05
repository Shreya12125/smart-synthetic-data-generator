import streamlit as st
import pandas as pd
from sdv.tabular import GaussianCopula
import matplotlib.pyplot as plt
import io
import re

# ---------- Page config ----------
st.set_page_config(page_title="Dual Synthetic Data Generator", page_icon="🧠", layout="wide")

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Controls")
uploaded_file_a = st.sidebar.file_uploader("📂 Upload Dataset A", type=["csv"])
uploaded_file_b = st.sidebar.file_uploader("📂 Upload Dataset B", type=["csv"])
num_rows = st.sidebar.number_input("📌 Rows to generate", 10, 10_000, 100, step=10)
gen_a = st.sidebar.button("🚀 Generate for Dataset A", key="gen_a", use_container_width=True)
gen_b = st.sidebar.button("🚀 Generate for Dataset B", key="gen_b", use_container_width=True)

with st.sidebar.expander("ℹ️ About this app"):
    st.markdown("Generate **privacy-friendly synthetic data** from two uploaded datasets. "
                "Uses [SDV](https://sdv.dev/) GaussianCopula model.")

# ---------- Helper: PII Detection ----------
def detect_pii_column(col, series):
    pii_tokens = ["name", "email", "phone", "mobile", "contact", "ssn", "dob", "address", "card"]
    if any(tok in col.lower() for tok in pii_tokens):
        return True
    sample = series.dropna().astype(str).head(10).tolist()
    patterns = [
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",  # Email
        r"\d{3}[-.\s]?\d{2}[-.\s]?\d{4}",                   # SSN
        r"\b\d{10,16}\b",                                   # Phone/Card
    ]
    return any(re.search(pat, val) for val in sample for pat in patterns)

# ---------- Main ----------
st.title("🧠 Dual Smart Synthetic Data Generator")

# ---------- Process Dataset A ----------
if uploaded_file_a:
    df_a = pd.read_csv(uploaded_file_a)
    synthetic_a = None

    st.header("📁 Dataset A")

    if gen_a:
        with st.spinner("Training model for Dataset A..."):
            model_a = GaussianCopula()
            model_a.fit(df_a)
            synthetic_a = model_a.sample(num_rows)
        st.success("Synthetic data for Dataset A generated!")

    tabs_a = st.tabs(["📋 Overview A", "📑 Original A", "🧪 Synthetic A", "📊 Charts A"])

    with tabs_a[0]:
        st.subheader("📋 Dataset A Summary")
        with st.expander("🔎 Quick Stats"):
            st.write(f"**Rows:** {df_a.shape[0]} | **Columns:** {df_a.shape[1]}")
            st.metric("Missing cells", int(df_a.isna().sum().sum()))
            st.dataframe(df_a.describe(include="all").T)

        with st.expander("🔐 PII Scan"):
            pii_cols = [col for col in df_a.columns if detect_pii_column(col, df_a[col])]
            if pii_cols:
                st.warning("⚠️ PII columns: " + ", ".join(pii_cols))
            else:
                st.success("✅ No obvious PII detected.")

    with tabs_a[1]:
        st.subheader("📑 Dataset A - Original Data")
        st.dataframe(df_a, use_container_width=True, height=400)

    with tabs_a[2]:
        st.subheader("🧪 Synthetic Data A")
        if synthetic_a is not None:
            st.dataframe(synthetic_a, use_container_width=True, height=400)
            buf_a = io.StringIO()
            synthetic_a.to_csv(buf_a, index=False)
            st.download_button("📥 Download Synthetic A", buf_a.getvalue(), "synthetic_a.csv", "text/csv", use_container_width=True)
        else:
            st.info("Click 'Generate for Dataset A' to view synthetic data.")

    with tabs_a[3]:
        st.subheader("📊 Dataset A - Distribution Comparison")
        if synthetic_a is not None:
            num_cols = df_a.select_dtypes("number").columns
            if num_cols.empty:
                st.info("No numeric columns to chart.")
            else:
                for col in num_cols:
                    fig, ax = plt.subplots()
                    df_a[col].plot(kind="kde", ax=ax, label="Original")
                    synthetic_a[col].plot(kind="kde", ax=ax, label="Synthetic")
                    ax.set_title(f"Distribution • {col}")
                    ax.legend()
                    st.pyplot(fig, use_container_width=True)

# ---------- Process Dataset B ----------
if uploaded_file_b:
    df_b = pd.read_csv(uploaded_file_b)
    synthetic_b = None

    st.header("📁 Dataset B")

    if gen_b:
        with st.spinner("Training model for Dataset B..."):
            model_b = GaussianCopula()
            model_b.fit(df_b)
            synthetic_b = model_b.sample(num_rows)
        st.success("Synthetic data for Dataset B generated!")

    tabs_b = st.tabs(["📋 Overview B", "📑 Original B", "🧪 Synthetic B", "📊 Charts B"])

    with tabs_b[0]:
        st.subheader("📋 Dataset B Summary")
        with st.expander("🔎 Quick Stats"):
            st.write(f"**Rows:** {df_b.shape[0]} | **Columns:** {df_b.shape[1]}")
            st.metric("Missing cells", int(df_b.isna().sum().sum()))
            st.dataframe(df_b.describe(include="all").T)

        with st.expander("🔐 PII Scan"):
            pii_cols = [col for col in df_b.columns if detect_pii_column(col, df_b[col])]
            if pii_cols:
                st.warning("⚠️ PII columns: " + ", ".join(pii_cols))
            else:
                st.success("✅ No obvious PII detected.")

    with tabs_b[1]:
        st.subheader("📑 Dataset B - Original Data")
        st.dataframe(df_b, use_container_width=True, height=400)

    with tabs_b[2]:
        st.subheader("🧪 Synthetic Data B")
        if synthetic_b is not None:
            st.dataframe(synthetic_b, use_container_width=True, height=400)
            buf_b = io.StringIO()
            synthetic_b.to_csv(buf_b, index=False)
            st.download_button("📥 Download Synthetic B", buf_b.getvalue(), "synthetic_b.csv", "text/csv", use_container_width=True)
        else:
            st.info("Click 'Generate for Dataset B' to view synthetic data.")

    with tabs_b[3]:
        st.subheader("📊 Dataset B - Distribution Comparison")
        if synthetic_b is not None:
            num_cols = df_b.select_dtypes("number").columns
            if num_cols.empty:
                st.info("No numeric columns to chart.")
            else:
                for col in num_cols:
                    fig, ax = plt.subplots()
                    df_b[col].plot(kind="kde", ax=ax, label="Original")
                    synthetic_b[col].plot(kind="kde", ax=ax, label="Synthetic")
                    ax.set_title(f"Distribution • {col}")
                    ax.legend()
                    st.pyplot(fig, use_container_width=True)

# ---------- Final Note ----------
if not uploaded_file_a and not uploaded_file_b:
    st.info("👈 Upload one or two CSV files in the sidebar to get started.")
