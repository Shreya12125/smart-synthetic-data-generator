# Smart Synthetic Data Generator

A Streamlit web app to generate synthetic (fake but realistic) data from any CSV using Generative AI — built with SDV’s GaussianCopula model.

> Great for privacy-safe ML training when original data is limited or sensitive.

---

## Features

- Upload CSV (supports 2 datasets: A and B)
- Generate synthetic tabular data using `sdv.tabular.GaussianCopula`
- Compare distributions: Original vs. Synthetic (KDE plots)
- Automatic PII column detection
- Dataset summary stats & schema
- Download synthetic data
- Clean, responsive UI built with Streamlit

---

## How to Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/smart-synthetic-data-generator.git
cd smart-synthetic-data-generator
