import streamlit as st
import pandas as pd
from io import BytesIO
from gao_audit_v3 import run_audit_dataframe, write_excel_report

st.set_page_config(page_title="GAO Schedule Auditor v3", layout="wide", page_icon="📘")

st.title("📘 GAO Schedule Quality Auditor v3")
st.caption("Upload your MS Project Excel export to generate a full GAO-compliant audit report.")

uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])

# Sidebar thresholds
st.sidebar.header("⚙️ Configuration")
