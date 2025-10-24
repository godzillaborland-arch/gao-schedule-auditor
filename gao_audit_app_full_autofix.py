import re
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

# Uses your existing audit logic if available
try:
    from gao_audit import run_audit as RUN_GAO_AUDIT  # expects a function that accepts a DataFrame
except Exception:
    RUN_GAO_AUDIT = None


# -----------------------------
# Config & column name aliases
# -----------------------------
COLUMN_ALIASES = {
    "Task Name": ["Task Name", "Name", "Activity Name", "Task"],
    "Start": ["Start", "Start_Date", "Start Date", "Begin"],
    "Finish": ["Finish", "Finish_Date", "Finish Date", "End"],
    "Duration": ["Duration", "Duration (minutes)", "Duration_Minutes"],
    "Predecessors": ["Predecessors", "Pred", "Predecessor", "Predecessor(s)"],
    "Successors": ["Successors", "Successor", "Successor(s)"],
    "Percent Complete": ["Percent Complete", "Percent_Complete", "% Complete", "Pct Complete"],
    "Baseline Finish": ["Baseline Finish", "Baseline_Finish", "BL Finish", "BL_Finish"],
    "Total Slack": ["Total Slack", "Total_Slack", "Slack", "TotalSlack"],
    "UID": ["UID", "Unique ID", "Unique_ID", "UniqueID", "Task UID", "ID"],
    "Summary": ["Summary"],
    "Milestone": ["Milestone", "Is Milestone", "Is_Milestone"],
    "Resource Names": ["Resource Names", "ResourceNames", "Resources"],
}

REQUIRED_FOR_VALIDATION = [
    "Task Name", "Start", "Finish", "Duration", "Predecessors", "Percent Complete"
]


# -----------------------------
# Helpers
# -----------------------------
def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column found from candidates (case-insensitive), else None."""
    colmap = {c.lower().strip(): c for c in df.columns}
    for c in candidates:
        key = c.lower().strip()
        if key in colmap:
            return colmap[key]
    return None


def apply_aliases(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    """
    Renames columns to the canonical names defined in COLUMN_ALIASES.
    Returns: (renamed_df, issues_list, rename_map)
    """
    issues = []
    rename_map = {}
    df_cols_lower = {c.lower().strip(): c for c in df.columns}

    for canonical, aliases in COLUMN_ALIASES.items():
        found = _first_present(df, aliases)
        if found and found != canonical:
            rename_map[found] = canonical

    # Perform rename
    if rename_map:
        df = df.rename(columns=rename_map)

    # Check missing required columns
    for col in REQUIRED_FOR_VALIDATION:
        if col not in df.columns:
            issues.append(f"❌ Missing required column: '{col}'")

    return df, issues, rename_map


def clean_duration_series(s: pd.Series) -> tuple[pd.Series, int]:
    """
    Convert Duration column to numeric (days). Silently auto-fixes:
    - '1638 days' -> 1638
    - '10 d', '15 weeks', etc. -> best effort extract number
    - 'NA', '—', '' -> 0
    Returns: (cleaned_series, count_autofixed)
    """
    autofixed = 0
    cleaned = []

    for v in s.fillna(""):
        orig = v
        if isinstance(v, (int, float)):
            cleaned.append(float(v))
            continue

        text = str(v).strip().lower()
        if text == "" or text in {"na", "n/a", "none", "-", "—"}:
            cleaned.append(0.0)
            if text != "":
                autofixed += 1
            continue

        # Extract first numeric token
        m = re.search(r"([+-]?\d+(\.\d+)?)", text)
        if m:
            num = float(m.group(1))
            # if text mentions 'week' scale rough weeks->days; if 'day' keep; if 'hour/min', let audit handle later if needed
            if "week" in text:
                num *= 7.0
            # any text beyond pure number counts as autofix
            if m.group(0) != text:
                autofixed += 1
            cleaned.append(num)
        else:
            # no number at all -> set 0, count as autofix
            cleaned.append(0.0)
            autofixed += 1

    return pd.Series(cleaned, index=s.index, dtype=float), autofixed


def validate_and_autofix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Apply alias mapping, auto-fix durations, and compute a validation summary table.
    Returns: (df_clean, summary_df, issues_list)
    """
    issues = []
    df1, alias_issues, rename_map = apply_aliases(df)
    issues.extend(alias_issues)

    # validation counts
    counts = {
        "Missing Required Columns": 0,
        "Blank Task Names": 0,
        "Invalid Start/Finish Dates": 0,
        "Duration Auto-Fixed": 0,
        "Empty Baseline Finish": 0,
        "Blank Predecessors": 0,
        "Blank Successors": 0,
    }

    # Missing columns -> summary count
    counts["Missing Required Columns"] = sum(1 for col in REQUIRED_FOR_VALIDATION if col not in df1.columns)

    # Blank names
    if "Task Name" in df1.columns:
        counts["Blank Task Names"] = int(df1["Task Name"].isna().sum())

    # Dates
    bad_dates = 0
    for dc in ("Start", "Finish"):
        if dc in df1.columns:
            try:
                _ = pd.to_datetime(df1[dc], errors="coerce")
                bad_dates += int(_.isna().sum())
            except Exception:
                bad_dates += len(df1)
    counts["Invalid Start/Finish Dates"] = bad_dates

    # Duration
    if "Duration" in df1.columns:
        df1["Duration"] = df1["Duration"].astype(object)  # keep strings intact for cleaning
        df1["Duration"], fixed_count = clean_duration_series(df1["Duration"])
        counts["Duration Auto-Fixed"] = int(fixed_count)
    else:
        issues.append("❌ Missing 'Duration' after alias mapping.")

    # Baseline Finish
    if "Baseline Finish" in df1.columns:
        counts["Empty Baseline Finish"] = int(df1["Baseline Finish"].isna().sum())
    # Logic blanks
    if "Predecessors" in df1.columns:
        counts["Blank Predecessors"] = int(df1["Predecessors"].astype(str).str.strip().eq("").sum())
    if "Successors" in df1.columns:
        counts["Blank Successors"] = int(df1["Successors"].astype(str).str.strip().eq("").sum())

    # Build summary dataframe
    summary = pd.DataFrame(
        {
            "Check": list(counts.keys()),
            "Detected Issues": list(counts.values()),
        }
    )

    # Friendly hint line for renames
    if rename_map:
        for old, new in rename_map.items():
            issues.append(f"ℹ Column renamed automatically: '{old}' → '{new}'")

    return df1, summary, issues


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="GAO Schedule Quality Auditor — Autofix", layout="wide", page_icon="📘")
st.title("GAO Schedule Quality Auditor — Autofix")
st.caption("Auto-detects column names, silently cleans Duration text, and shows data quality summary before the audit.")

uploaded = st.file_uploader("📤 Upload Microsoft Project Excel (.xlsx)", type=["xlsx"])

if uploaded:
    try:
        # Offer sheet selection if multiple sheets
        xls = pd.ExcelFile(uploaded)
        sheet = st.selectbox("Select worksheet", xls.sheet_names, index=0)
        raw_df = pd.read_excel(uploaded, sheet_name=sheet)
        st.success(f"Loaded: {sheet} ({len(raw_df):,} rows)")
    except Exception as e:
        st.error(f"❌ Failed to read Excel file: {e}")
        st.stop()

    # Validate + autofix
    df_clean, validation_summary, messages = validate_and_autofix(raw_df)

    st.subheader("📊 Data Validation Summary")
    st.dataframe(validation_summary, use_container_width=True)

    if messages:
        st.info("Notes:")
        for m in messages:
            st.write("•", m)

    # If truly missing required columns, stop (let users fix/re-upload)
    missing_required = validation_summary.loc[
        validation_summary["Check"] == "Missing Required Columns", "Detected Issues"
    ].iloc[0]
    if missing_required > 0:
        st.error("❌ Critical: Required columns missing. Please fix your Excel headers and re-upload.")
        st.stop()

    # Proceed to audit
    st.markdown("---")
    st.info("All critical checks passed. Running GAO audit… ⏳")

    if RUN_GAO_AUDIT is None:
        st.error("Could not import 'run_audit' from gao_audit.py. Please ensure it exists and is importable.")
        st.stop()

    try:
        results_df = RUN_GAO_AUDIT(df_clean)
    except Exception as e:
        st.error(f"❌ Audit failed: {e}")
        st.exception(e)
        st.stop()

    st.success("✅ Audit complete!")
    st.dataframe(results_df, use_container_width=True)

    # Download results
    out = BytesIO()
    results_df.to_excel(out, index=False)
    st.download_button(
        "⬇️ Download Full Audit Report",
        data=out.getvalue(),
        file_name="GAO_Schedule_Audit_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

else:
    st.info("Upload a Microsoft Project Excel export (.xlsx) to begin.")
