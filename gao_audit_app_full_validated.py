import streamlit as st
import pandas as pd
from io import BytesIO
from gao_audit import run_audit  # core audit logic

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="GAO Schedule Quality Auditor",
    layout="wide",
    page_icon="📘"
)

# --- Helper: Validation Function ---
def validate_schedule(df: pd.DataFrame) -> list[str]:
    """Check for missing or invalid data in the uploaded schedule."""
    issues = []
    required_cols = [
        "Task Name", "Start", "Finish", "Duration",
        "Predecessors", "Percent Complete"
    ]

    for col in required_cols:
        if col not in df.columns:
            issues.append(f"❌ Missing required column: '{col}'")

    if "Task Name" in df.columns and df["Task Name"].isna().any():
        issues.append("⚠ Some tasks are missing names.")

    for date_col in ["Start", "Finish"]:
        if date_col in df.columns:
            try:
                pd.to_datetime(df[date_col])
            except Exception:
                issues.append(f"⚠ Invalid date format in '{date_col}' column.")

    if "Duration" in df.columns:
        non_numeric = df["Duration"].apply(lambda x: not str(x).replace('.', '', 1).isdigit()).sum()
        if non_numeric > 0:
            issues.append(f"⚠ {non_numeric} rows have non-numeric durations.")

    if "Baseline Finish" in df.columns and df["Baseline Finish"].isna().all():
        issues.append("⚠ Baseline Finish column is empty (no baseline data).")

    return issues

# --- App Layout ---
st.title("GAO Schedule Quality Auditor")
st.caption("Automated GAO-style Schedule Health and Logic Analysis")

uploaded_file = st.file_uploader("📤 Upload your Microsoft Project Excel export (.xlsx)", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("✅ File uploaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to read Excel file: {e}")
        st.stop()

      # --- Enhanced Data Validation Summary ---
    problems = validate_schedule(df)

    # Build validation summary counts
    summary = {
        "Check": [
            "Missing Required Columns",
            "Blank Task Names",
            "Invalid Dates",
            "Non-Numeric Durations",
            "Empty Baseline Finish"
        ],
        "Detected Issues": [0, 0, 0, 0, 0]
    }

    # Count problems per type
    for p in problems:
        if "Missing required column" in p:
            summary["Detected Issues"][0] += 1
        elif "missing names" in p:
            summary["Detected Issues"][1] += 1
        elif "Invalid date format" in p:
            summary["Detected Issues"][2] += 1
        elif "non-numeric durations" in p:
            summary["Detected Issues"][3] += 1
        elif "Baseline Finish" in p:
            summary["Detected Issues"][4] += 1

    # Display summary table
    st.subheader("📊 Data Validation Summary")
    summary_df = pd.DataFrame(summary)
    st.dataframe(summary_df, use_container_width=True)

    if problems:
        st.error("⚠ Data validation found the following issues:")
        for p in problems:
            st.write("-", p)
        st.stop()
    else:
        st.success("✅ All validation checks passed. Proceeding with GAO audit...")


    # --- Proceed with audit ---
    st.info("All checks passed. Running GAO audit... ⏳")
    results_df = run_audit(df)

    st.success("✅ Audit complete!")
    st.dataframe(results_df)

    # --- Download section ---
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
st.caption("© 2025 GAO Schedule Quality Auditor | Quantum View Point")
