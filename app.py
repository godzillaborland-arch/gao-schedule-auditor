import streamlit as st
import pandas as pd
from gao_audit import run_audit, write_report, load_schedule
import tempfile, os

st.set_page_config(page_title="GAO Schedule Auditor", page_icon="📊", layout="wide")
st.title("📘 GAO Schedule Quality Auditor")
st.markdown("Upload a Microsoft Project **Excel export (.xlsx)** to generate a GAO-style schedule quality report.")

uploaded_file = st.file_uploader("Upload your schedule file", type=["xlsx"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    with st.spinner("Running audit... please wait ⏳"):
        try:
            df = load_schedule(tmp_path)  # auto-detect sheet or use Task_Table1
            audit = run_audit(df)
            output_path = os.path.splitext(tmp_path)[0] + "_Report.xlsx"
            score, health = write_report(audit, output_path)

            st.success("✅ Audit completed successfully!")
            st.metric("Schedule Health Score", f"{score}/100")
            st.write("**Project Health:**", health)

            st.subheader("Summary")
            st.dataframe(audit["Summary"])

            with open(output_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Excel Report",
                    f,
                    file_name="GAO_Schedule_Audit_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
