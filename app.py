import streamlit as st
import pandas as pd
from io import BytesIO
from gao_audit_v3 import run_audit  # main GAO audit logic

st.set_page_config(page_title="GAO Schedule Quality Auditor v3", layout="wide")

st.title("📘 GAO Schedule Quality Auditor v3")
st.write("Upload your MS Project Excel export to generate a full GAO-compliant audit report.")

uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"], help="Limit 200MB per file • XLSX")

if uploaded_file:
    st.info(f"📄 {uploaded_file.name}")
    run_clicked = st.button("▶️ Run Audit")

    if run_clicked:
        try:
            with st.spinner("Running GAO audit..."):
                df = pd.read_excel(uploaded_file)
                results_df = run_audit(df)
                st.success("✅ Audit completed successfully!")
                st.dataframe(results_df, use_container_width=True)

                towrite = BytesIO()
                results_df.to_excel(towrite, index=False)
                towrite.seek(0)
                st.download_button(
                    label="📥 Download Audit Report (.xlsx)",
                    data=towrite,
                    file_name="GAO_Schedule_Audit_Report_v3.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        except Exception as e:
            st.error(f"❌ Audit failed: {e}")
else:
    st.info("👆 Upload a valid Microsoft Project schedule (.xlsx) and click **Run Audit**.")

st.markdown("---")
st.caption("© 2025 GAO Schedule Quality Auditor | Built with ❤️ using Streamlit")
