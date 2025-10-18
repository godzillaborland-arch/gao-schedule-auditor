import streamlit as st
import pandas as pd
from io import BytesIO
from gao_audit import run_audit  # your main logic

# --- Page Config ---
st.set_page_config(
    page_title="GAO Schedule Quality Auditor",
    layout="wide",
    page_icon="📘"
)
from pathlib import Path

# --- Display Quantum View Point Logo ---
logo_path = Path("assets/gao_logo.png")
if logo_path.exists():
    st.image(str(logo_path), width=200)
else:
    st.warning("Logo not found. Check path or filename.")


# --- Custom CSS: GAO theme fallback ---
st.markdown("""
    <style>
    /* Backgrounds and Fonts */
    .main {
        background-color: #F9FBFC;
    }
    h1, h2, h3, h4 {
        color: #003366;
        font-weight: 700;
    }
    /* Buttons */
    .stButton>button {
        background-color: #003366;
        color: white;
        border-radius: 6px;
        border: none;
        font-weight: bold;
        padding: 0.6em 1.2em;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #00509E;
    }
    /* Metrics cards */
    div[data-testid="stMetricValue"] {
        color: #003366;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/35/US-GovernmentAccountabilityOffice-Logo.svg/2560px-US-GovernmentAccountabilityOffice-Logo.svg.png",
         width=220)
st.title("GAO Schedule Quality Auditor")
st.caption("Automated Schedule Health and Logic Analysis – Powered by Streamlit + Excel")

st.markdown("---")

# --- File Upload ---
st.subheader("📤 Upload your Microsoft Project Export (.xlsx)")
uploaded_file = st.file_uploader(
    "Drag and drop your schedule file or click to browse",
    type=["xlsx"]
)

if uploaded_file:
    with st.spinner("🔍 Analyzing schedule... please wait"):
        # Run your GAO audit logic
        results_df = run_audit(uploaded_file)

    st.success("✅ Analysis complete!")
    st.dataframe(results_df)

    # Download results
    output = BytesIO()
    results_df.to_excel(output, index=False)
    st.download_button(
        label="⬇️ Download Full Audit Report",
        data=output.getvalue(),
        file_name="GAO_Schedule_Audit_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("Please upload a valid Microsoft Project Excel export (.xlsx) to begin.")

st.markdown("---")
st.caption("© 2025 GAO Schedule Quality Auditor | Built with ❤️ using Streamlit")
