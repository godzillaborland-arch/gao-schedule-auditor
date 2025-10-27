# gao_audit_app_full_autofix.py
# Streamlit UI that wraps the shared GAO core: upload Excel, validate+autofix, run audit,
# show summary + download report. Includes badge-style validation table.

from io import BytesIO
import pandas as pd
import streamlit as st
from gao_core import validate_and_autofix, run_gao_audit, DEFAULTS

st.set_page_config(page_title="GAO Schedule Quality Auditor — AutoFix", layout="wide", page_icon="📘")
st.title("GAO Schedule Quality Auditor — AutoFix")
st.caption("Auto-detects column names, silently cleans Duration text, and shows a data quality summary before the audit.")

# Sidebar config
st.sidebar.header("Configuration")
sheet_name = st.sidebar.text_input("Sheet name (optional)", value="")
excessive_slack_days = st.sidebar.number_input("Excessive Slack Threshold (days)", 1, 3650, DEFAULTS["EXCESSIVE_SLACK_DAYS"])
long_duration_days = st.sidebar.number_input("Long Duration (days)", 1, 3650, DEFAULTS["LONG_DURATION_DAYS"])
short_duration_min = st.sidebar.number_input("Short Duration (minutes)", 1, 14400, DEFAULTS["SHORT_DURATION_MIN"])
lead_lag_abs_limit = st.sidebar.number_input("Lead/Lag absolute limit", 0, 3650, DEFAULTS["LEAD_LAG_ABS_LIMIT"])

uploaded = st.file_uploader("📤 Upload Microsoft Project Excel (.xlsx)", type=["xlsx"])

def _badge(val: int) -> str:
    if val == 0:
        return "🟢"
    if val <= 10:
        return "🟡"
    return "🔴"

if uploaded:
    try:
        # Allow choosing sheet if desired
        if sheet_name.strip():
            df_clean, vsummary, msgs = validate_and_autofix(uploaded, sheet_name=sheet_name.strip())
        else:
            df_clean, vsummary, msgs = validate_and_autofix(uploaded)
        st.success(f"Loaded file. Rows: {len(df_clean):,}")
    except Exception as e:
        st.error(f"❌ Failed to read Excel file: {e}")
        st.stop()

    # Decorate validation with badges
    vdisp = vsummary.copy()
    vdisp["Status"] = vdisp["Detected Issues"].apply(_badge)
    vdisp = vdisp[["Status", "Check", "Detected Issues"]]

    st.subheader("📊 Data Validation Summary")
    st.dataframe(vdisp, use_container_width=True)
    if msgs:
        with st.expander("Notes & Auto-fixes", expanded=False):
            for m in msgs:
                st.write("•", m)

    # Stop if critically missing required columns
    missing_row = vsummary.loc[vsummary["Check"] == "Missing Required Columns", "Detected Issues"]
    if not missing_row.empty and missing_row.iloc[0] > 0:
        st.error("❌ Critical: Required columns are missing. Please fix your Excel headers and re-upload.")
        st.stop()

    st.markdown("---")
    st.info("All critical checks passed. Running GAO audit… ⏳")

    try:
        summary_df, details = run_gao_audit(
            df_clean,
            excessive_slack_days=excessive_slack_days,
            long_duration_days=long_duration_days,
            short_duration_min=short_duration_min,
            lead_lag_abs_limit=lead_lag_abs_limit,
        )
    except Exception as e:
        st.error(f"❌ Audit failed: {e}")
        st.exception(e)
        st.stop()

    st.success("✅ Audit complete!")
    col1, col2 = st.columns([2, 3])
    with col1:
        st.subheader("Summary Metrics")
        st.dataframe(summary_df, use_container_width=True)
    with col2:
        st.subheader("Details (top 1k rows per sheet)")
        for name, df in details.items():
            st.markdown(f"**{name}**")
            st.dataframe(df.head(1000), use_container_width=True)

    # Download full report
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        summary_df.to_excel(w, sheet_name="Summary", index=False)
        for name, df in details.items():
            df.to_excel(w, sheet_name=name[:31], index=False)
    st.download_button(
        "⬇️ Download Full Audit Report",
        data=out.getvalue(),
        file_name="GAO_Schedule_Audit_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

else:
    st.info("Upload a Microsoft Project Excel export (.xlsx) to begin.")
