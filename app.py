import streamlit as st
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="GAO Schedule Quality Auditor v3", layout="wide")

st.title("📘 GAO Schedule Quality Auditor v3")
st.write("Upload your MS Project Excel export to generate a full GAO-compliant audit report.")

uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"], help="Limit 200MB per file • XLSX")

def run_audit(df):
    summary = []
    if "Total_Slack" not in df.columns:
        summary.append({"Metric": "Slack Status", "Value": "No Total_Slack column found"})
        return pd.DataFrame(summary)

    df["Total_Slack"] = pd.to_numeric(df["Total_Slack"], errors="coerce")

    negative_slack = len(df[df["Total_Slack"] < 0])
    excessive_slack = len(df[df["Total_Slack"] > 60])
    missing_pred = len(df[df["Predecessors"].isna() | (df["Predecessors"] == "")])
    missing_succ = len(df[df["Successors"].isna() | (df["Successors"] == "")])
    constraints = len(df[df["Constraint_Type"].notna() & (df["Constraint_Type"] != "") & (df["Constraint_Type"].str.lower() != "as soon as possible")])

    summary.extend([
        {"Metric": "Negative Slack", "Value": negative_slack},
        {"Metric": "Excessive Slack (>60d)", "Value": excessive_slack},
        {"Metric": "Missing Predecessors", "Value": missing_pred},
        {"Metric": "Missing Successors", "Value": missing_succ},
        {"Metric": "Constraints", "Value": constraints}
    ])
    return pd.DataFrame(summary)


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
