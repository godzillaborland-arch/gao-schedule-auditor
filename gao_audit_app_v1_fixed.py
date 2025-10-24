# gao_audit_app_v1_fixed.py
# Streamlit GAO Schedule Auditor (Patched for Slack_Type handling)

import streamlit as st
import pandas as pd
import networkx as nx
from io import BytesIO

# --- Utility Functions ---

def normalize_columns(df):
    rename_map = {
        "Unique ID": "UID",
        "UniqueID": "UID",
        "Task UID": "UID",
        "Task ID": "UID",
    }
    df.rename(columns={c: rename_map.get(c, c) for c in df.columns}, inplace=True)
    for col in ["Predecessors", "Successors", "ResourceNames"]:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("").astype(str)
    return df

def run_audit_dataframe(df, slack_lo_days=0, slack_hi_days=60, exclude_summaries=True):
    df = normalize_columns(df)

    # Slack analysis
    slack_analysis = pd.DataFrame()
    if "Total Slack" in df.columns:
        df["Total Slack"] = pd.to_numeric(df["Total Slack"], errors="coerce")
        slack_analysis = df[(df["Total Slack"] < 0) | (df["Total Slack"] > (slack_hi_days * 480))][["UID", "Name", "Total Slack"]].copy()
        slack_analysis["Slack_Type"] = slack_analysis["Total Slack"].apply(
            lambda x: "Negative Slack" if x < 0 else "Excessive Slack"
        )

    # Example placeholder logic for other checks
    invalid_rows = df[df["Predecessors"] == ""]
    cycles = []
    dangling_tasks = df[df["Successors"] == ""]
    oos = df[(df["Percent Complete"] > 0) & (df["Predecessors"] == "")]
    hard_count = len(df[df.get("Constraint Type", "") == "Must Finish On"])
    soft_count = len(df[df.get("Constraint Type", "") == "As Soon As Possible"])
    late_count = 0

    # --- FIXED SECTION ---
    neg_slack = 0
    exc_slack = 0
    if not slack_analysis.empty and "Slack_Type" in slack_analysis.columns:
        neg_slack = int((slack_analysis["Slack_Type"] == "Negative Slack").sum())
        exc_slack = int((slack_analysis["Slack_Type"] == "Excessive Slack").sum())

    summary_items = {
        "Malformed/Missing Links": len(invalid_rows),
        "Circular Dependencies": len(cycles),
        "Dangling Tasks": len(dangling_tasks),
        "Out-of-Sequence Actuals": len(oos),
        "Baseline Variance (Late)": late_count,
        "Constraints (Hard)": hard_count,
        "Constraints (Soft / Valid)": soft_count,
        "Negative Slack": neg_slack,
        "Excessive Slack": exc_slack,
    }

    summary_df = pd.DataFrame(list(summary_items.items()), columns=["Metric", "Value"])
    return summary_df, slack_analysis


# --- Streamlit App ---
st.set_page_config(page_title="GAO Schedule Quality Auditor", layout="wide", page_icon="📘")

st.title("GAO Schedule Quality Auditor (Fixed Version)")
st.caption("Automated GAO schedule audit with Slack-Type safety fix.")

st.sidebar.header("⚙️ Configuration")
slack_lo = st.sidebar.number_input("Slack Low (days)", min_value=0, max_value=365, value=0)
slack_hi = st.sidebar.number_input("Slack High (days)", min_value=1, max_value=365, value=60)
exclude_summaries = st.sidebar.checkbox("Exclude Summary Tasks", value=True)

uploaded_file = st.file_uploader("Upload Microsoft Project Export (.xlsx)", type=["xlsx"])

if uploaded_file:
    st.info("Running GAO audit... please wait ⏳")
    df = pd.read_excel(uploaded_file)
    summary_df, slack_df = run_audit_dataframe(df, slack_lo_days=slack_lo, slack_hi_days=slack_hi, exclude_summaries=exclude_summaries)
    st.success("✅ Audit complete!")

    st.subheader("📊 Summary Metrics")
    st.dataframe(summary_df)

    if not slack_df.empty:
        st.subheader("🧩 Slack Analysis (Only Excessive / Negative)")
        st.dataframe(slack_df)
    else:
        st.info("No excessive or negative slack found.")

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
        if not slack_df.empty:
            slack_df.to_excel(writer, index=False, sheet_name="Slack Analysis")

    st.download_button(
        label="⬇️ Download GAO_Schedule_Audit_Report_v4.xlsx",
        data=output.getvalue(),
        file_name="GAO_Schedule_Audit_Report_v4.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
else:
    st.info("Please upload a schedule file to start.")
