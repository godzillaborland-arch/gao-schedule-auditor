# --- GAO Schedule Quality Auditor (Full) ---
# Streamlit UI + complete audit logic with robust column detection.

import io
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import networkx as nx
from collections import Counter


# ----------------------------
# Helpers: parsing & coercion
# ----------------------------

DUR_UNIT_MAP = {
    "m": 1 / 60.0,   # minutes -> hours
    "h": 1.0,        # hours  (treat as 1)
    "d": 8.0,        # "days" as 8h
    "w": 40.0,       # weeks  as 5*8h
    "y": 2080.0,     # years  as 52*40h
}

def _clean_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def parse_duration_to_days(val) -> float:
    """
    Accepts things like: '425d', '196 days', '0d', '90', 120 (minutes?), '2w'.
    Converts to **days** assuming 8h/day if only minutes/hours are given.
    If it's plainly numeric, we assume it's already in minutes (MSP export),
    convert to days by minutes/480.
    """
    if pd.isna(val):
        return np.nan

    # numeric => assume minutes, convert to days by 480
    if isinstance(val, (int, float)):
        try:
            return float(val) / 480.0
        except Exception:
            return np.nan

    s = str(val).strip().lower()

    # "### days"
    m = re.match(r"^\s*([+-]?\d+(\.\d+)?)\s*days?\s*$", s)
    if m:
        return float(m.group(1))

    # e.g. "425d", "2w", "60m", "8h"
    m = re.match(r"^\s*([+-]?\d+(\.\d+)?)([mhdwy])\s*$", s)
    if m:
        num = float(m.group(1))
        unit = m.group(3)
        if unit in DUR_UNIT_MAP:
            hours = num * DUR_UNIT_MAP[unit]
            # If unit already in days:
            if unit == "d":
                return num
            else:
                return hours / 8.0

    # plain number as string => assume minutes
    try:
        return float(s) / 480.0
    except Exception:
        return np.nan


def parse_dependency_tokens(text: str) -> Tuple[List[Tuple[int, str, str]], List[str]]:
    """
    Accepts strings like: '101FS, 102SS+3d, 103FF-2w'
    Returns:
        tokens = [(pred_id:int, dep_type:str, offset:str or None), ...]
        invalid = [raw_token, ...]
    """
    tokens, invalid = [], []
    if not isinstance(text, str) or not text.strip():
        return tokens, invalid

    for raw in re.split(r"[,\s]+", text.strip()):
        if not raw:
            continue
        m = re.match(r"^(\d+)([FS]{1,2})?([+-]?\d+[mhdwy])?$", raw, re.IGNORECASE)
        if m:
            pred = int(m.group(1))
            typ = (m.group(2) or "FS").upper()
            off = m.group(3)
            tokens.append((pred, typ, off))
        else:
            invalid.append(raw)
    return tokens, invalid


# -----------------------------------
# Normalize & harmonize input columns
# -----------------------------------

RENAME_CANDIDATES: Dict[str, List[str]] = {
    "UID": ["UID", "Unique ID", "Unique_ID", "UniqueID", "Task UID", "ID"],
    "Name": ["Name", "Task Name", "Task_Name"],
    "Predecessors": ["Predecessors", "Pred", "Predecessor", "Predecessor(s)"],
    "Successors": ["Successors", "Successor", "Successor(s)"],
    "ResourceNames": ["Resource Names", "ResourceNames", "Resources"],
    "Summary": ["Summary"],
    "Milestone": ["Milestone", "Is Milestone", "Is_Milestone"],
    "TotalSlack": ["Total Slack", "Total_Slack", "Slack", "TotalSlack"],
    "PercentComplete": ["Percent Complete", "Percent_Complete", "% Complete", "Pct Complete"],
    "BaselineStart": ["Baseline Start", "Baseline_Start", "BL Start", "BL_Start"],
    "BaselineFinish": ["Baseline Finish", "Baseline_Finish", "BL Finish", "BL_Finish"],
    "Start": ["Start", "Start_Date", "Scheduled Start"],
    "Finish": ["Finish", "Finish_Date", "Scheduled Finish"],
    "Duration": ["Duration", "Duration (minutes)", "Duration_Minutes"],
    "ConstraintType": ["Constraint Type", "Constraint_Type", "ConstraintType"],
}

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    src_cols = list(df.columns)

    # Build rename map by first exact, then case-insensitive match
    for target, candidates in RENAME_CANDIDATES.items():
        hit = None
        for c in candidates:
            if c in df.columns:
                hit = c
                break
        if not hit:
            # case-insensitive
            lower = {c.lower(): c for c in df.columns}
            for c in candidates:
                if c.lower() in lower:
                    hit = lower[c.lower()]
                    break
        if hit:
            col_map[hit] = target

    if col_map:
        df = df.rename(columns=col_map)

    # Ensure column presence with safe defaults
    if "UID" not in df.columns:
        df["UID"] = range(1, len(df) + 1)

    if "Name" not in df.columns:
        df["Name"] = "(unnamed)"

    for c in ["Predecessors", "Successors", "ResourceNames"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
        else:
            df[c] = ""

    for c in ["Summary", "Milestone"]:
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)
        else:
            df[c] = False

    # Percent complete: coerce both "Percent Complete" and "Percent_Complete"
    if "PercentComplete" not in df.columns:
        # Try to find any percent-like column
        for cand in ["Percent Complete", "Percent_Complete", "% Complete", "Pct Complete"]:
            if cand in src_cols:
                df["PercentComplete"] = pd.to_numeric(df[cand], errors="coerce")
                break
    if "PercentComplete" not in df.columns:
        df["PercentComplete"] = 0.0
    # Normalize 0..1 if data looks like 0..1 floats
    if df["PercentComplete"].max() <= 1.0:
        df["PercentComplete"] = df["PercentComplete"] * 100.0
    df["PercentComplete"] = pd.to_numeric(df["PercentComplete"], errors="coerce").fillna(0.0)

    # Slack -> to days (float)
    if "TotalSlack" in df.columns:
        df["TotalSlackDays"] = df["TotalSlack"].apply(parse_duration_to_days)
    else:
        df["TotalSlackDays"] = np.nan

    # Duration -> minutes or days unify -> store minutes + a days helper
    if "Duration" in df.columns:
        # If it's numeric, assume minutes; if text, parse; then keep both
        def _dur_to_minutes(v):
            if pd.isna(v):
                return np.nan
            if isinstance(v, (int, float)):
                return float(v)
            # parse to days, then to minutes
            d = parse_duration_to_days(v)
            if pd.isna(d):
                return np.nan
            return d * 480.0

        df["DurationMin"] = df["Duration"].apply(_dur_to_minutes)
        df["DurationDays"] = df["DurationMin"] / 480.0
    else:
        df["DurationMin"] = np.nan
        df["DurationDays"] = np.nan

    # Baseline presence flags
    for c in ["BaselineStart", "BaselineFinish", "Start", "Finish"]:
        if c not in df.columns:
            df[c] = np.nan

    return df


# -----------------------------------
# Core GAO checks (configurable)
# -----------------------------------

DEFAULT_WEIGHTS = {
    "Malformed/Missing Links": 2.0,
    "Circular Dependencies": 4.0,
    "Dangling Tasks": 2.0,
    "Lead/Lag Warnings": 1.5,
    "Critical Path Logic Gaps": 1.0,
    "Negative Slack": 2.0,
    "Excessive Slack": 1.0,
    "Constraints": 1.0,
    "No Resources": 1.0,
    "Out-of-Sequence Actuals": 2.0,
    "No Baseline": 1.0,
    "Unrealistic Duration": 1.0,
    "Isolated Milestone": 1.0,
    "Low-Quality Name": 0.5,
    "Baseline Variance (Late)": 1.0,
}

SAFE_CONSTRAINTS = {
    "", "na", "n/a", "none", "as soon as possible", "asap", "as-soon-as-possible"
}


def run_audit_dataframe(
    df: pd.DataFrame,
    leadlag_abs_limit: int = 10,     # +/- units in m/h/d/w/y text (e.g. '10d' '4w')
    excessive_slack_days: float = 60,
    long_duration_days: float = 60,
    short_duration_minutes: float = 60,
    exclude_summaries: bool = True,
    weights: Dict[str, float] = None
) -> Dict[str, pd.DataFrame]:
    """
    Returns dict of dataframes:
      - summary (rows = metric,value)
      - multiple detail sheets
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    df = df.copy()
    N = len(df)

    # Optionally ignore summaries for some checks
    row_mask = ~df["Summary"] if exclude_summaries and "Summary" in df.columns else pd.Series([True] * N, index=df.index)
    working = df[row_mask].copy()

    # ----- Parse predecessors & collect dependency info -----
    valid_links = []      # (pred, task, type)
    invalid_rows = []     # malformed tokens
    lead_lag_warnings = []

    for _, row in working.iterrows():
        tokens, invalid = parse_dependency_tokens(row["Predecessors"])
        if invalid:
            invalid_rows.append({
                "UID": row["UID"], "Name": row["Name"],
                "Malformed_Tokens": ", ".join(invalid)
            })
        for pred_id, typ, off in tokens:
            if off:
                try:
                    # detect numeric portion regardless of unit, compare absolute
                    n = int(re.findall(r"[+-]?\d+", off)[0])
                    if abs(n) > leadlag_abs_limit:
                        lead_lag_warnings.append({"UID": row["UID"], "Name": row["Name"], "LeadLag": off})
                except Exception:
                    pass
            valid_links.append((pred_id, row["UID"], typ))

    # Graph & cycles
    G = nx.DiGraph()
    G.add_nodes_from(working["UID"].tolist())
    G.add_edges_from([(a, b) for a, b, _ in valid_links])

    try:
        cyc = list(nx.find_cycle(G, orientation="original"))
    except nx.NetworkXNoCycle:
        cyc = []
    cycles_df = pd.DataFrame(cyc, columns=["From", "To", "Type"]) if cyc else pd.DataFrame(columns=["From", "To", "Type"])

    # Dependency type counts
    dep_type_counts = Counter([t for _, _, t in valid_links])

    # Dangling = not connected at all
    all_ids = set(working["UID"].tolist())
    connected = {a for a, _, _ in valid_links} | {b for _, b, _ in valid_links}
    dangling_ids = list(all_ids - connected)
    dangling_df = working[working["UID"].isin(dangling_ids)][["UID", "Name"]]

    # CP logic gaps = tasks with no FS link in or out
    incoming_fs = {b for a, b, t in valid_links if t == "FS"}
    outgoing_fs = {a for a, b, t in valid_links if t == "FS"}
    no_fs = list(all_ids - (incoming_fs | outgoing_fs))
    cp_gaps_df = working[working["UID"].isin(no_fs)][["UID", "Name"]]

    # Slack checks
    slack_df = pd.DataFrame(columns=["UID", "Name", "SlackDays", "Slack_Type"])
    if "TotalSlackDays" in working.columns:
        s = pd.to_numeric(working["TotalSlackDays"], errors="coerce")
        neg = working[s < 0][["UID", "Name"]].copy()
        neg["SlackDays"] = s[s < 0]
        neg["Slack_Type"] = "Negative Slack"

        exc = working[s > excessive_slack_days][["UID", "Name"]].copy()
        exc["SlackDays"] = s[s > excessive_slack_days]
        exc["Slack_Type"] = "Excessive Slack"

        slack_df = pd.concat([neg, exc], ignore_index=True) if not neg.empty or not exc.empty else slack_df

    # Constraints (excluding ASAP + blank/NA)
    cons_df = pd.DataFrame(columns=["UID", "Name", "ConstraintType"])
    if "ConstraintType" in df.columns:
        col = df["ConstraintType"].astype(str).str.strip().str.lower()
        mask = ~col.isna()
        if mask.any():
            norm = col.where(mask, "")
            norm = norm.fillna("").str.strip()
            # exclude "as soon as possible" & empties
            keep = ~norm.isin(SAFE_CONSTRAINTS)
            cons_df = df[keep][["UID", "Name", "ConstraintType"]].copy()

    # No resources
    no_res_df = working[working["ResourceNames"].astype(str).str.strip() == ""][["UID", "Name"]]

    # Out-of-sequence actuals (simple heuristic):
    # if PercentComplete>0 and NO predecessors text -> likely started without logic
    oos_df = working[(pd.to_numeric(working["PercentComplete"], errors="coerce") > 0.0) &
                     (working["Predecessors"].astype(str).str.strip() == "")][["UID", "Name"]]

    # No baseline (either start or finish blank/NA)
    no_base_df = pd.DataFrame(columns=["UID", "Name"])
    if "BaselineStart" in df.columns and "BaselineFinish" in df.columns:
        mask = df["BaselineStart"].isna() | df["BaselineFinish"].isna() | \
               (df["BaselineStart"].astype(str).str.upper() == "NA") | (df["BaselineFinish"].astype(str).str.upper() == "NA")
        no_base_df = df[mask][["UID", "Name"]]

    # Unrealistic durations
    unreal_df = pd.DataFrame(columns=["UID", "Name", "DurationMin", "Flag"])
    if "DurationMin" in working.columns:
        # long
        long_mask = working["DurationDays"] > long_duration_days
        if long_mask.any():
            tmp = working[long_mask][["UID", "Name", "DurationMin"]].copy()
            tmp["Flag"] = "Unrealistic Duration: Long"
            unreal_df = pd.concat([unreal_df, tmp], ignore_index=True)
        # short
        short_mask = working["DurationMin"] < short_duration_minutes
        if short_mask.any():
            tmp = working[short_mask][["UID", "Name", "DurationMin"]].copy()
            tmp["Flag"] = "Unrealistic Duration: Short"
            unreal_df = pd.concat([unreal_df, tmp], ignore_index=True)

    # Isolated milestones = milestone and (no predecessors or no successors)
    iso_ms_df = working[(working["Milestone"]) &
                        ((working["Predecessors"].astype(str).str.strip() == "") |
                         (working["Successors"].astype(str).str.strip() == ""))][["UID", "Name"]]

    # Low-quality names (len<10) for non-summaries
    lq_names_df = working[working["Name"].astype(str).str.len() < 10][["UID", "Name"]]

    # Baseline variance (Late): if Finish & BaselineFinish present and Finish>BaselineFinish
    base_var_df = pd.DataFrame(columns=["UID", "Name", "BaselineFinish", "Finish", "VarianceDays"])
    try:
        if "Finish" in df.columns and "BaselineFinish" in df.columns:
            fin = pd.to_datetime(df["Finish"], errors="coerce")
            blf = pd.to_datetime(df["BaselineFinish"], errors="coerce")
            late_mask = (fin.notna()) & (blf.notna()) & (fin > blf)
            tmp = df[late_mask][["UID", "Name", "BaselineFinish", "Finish"]].copy()
            tmp["VarianceDays"] = (fin[late_mask] - blf[late_mask]).dt.days
            base_var_df = tmp
    except Exception:
        pass

    # Compose summary
    summary_items = {
        "Malformed/Missing Links": len(invalid_rows),
        "Circular Dependencies": len(cycles_df),
        "Dangling Tasks": len(dangling_df),
        "Lead/Lag Warnings": len(lead_lag_warnings),
        "Critical Path Logic Gaps": len(cp_gaps_df),
        "Negative Slack": int((slack_df["Slack_Type"] == "Negative Slack").sum()) if not slack_df.empty else 0,
        "Excessive Slack": int((slack_df["Slack_Type"] == "Excessive Slack").sum()) if not slack_df.empty else 0,
        "Constraints": len(cons_df),
        "No Resources": len(no_res_df),
        "Out-of-Sequence Actuals": len(oos_df),
        "No Baseline": len(no_base_df),
        "Unrealistic Duration": len(unreal_df),
        "Isolated Milestone": len(iso_ms_df),
        "Low-Quality Name": len(lq_names_df),
        "Baseline Variance (Late)": len(base_var_df),
    }
    summary_df = pd.DataFrame(list(summary_items.items()), columns=["Metric", "Value"])

    # Score
    total_tasks = max(1, len(working))
    penalty = 0.0
    for metric, value in summary_items.items():
        penalty += float(value) * weights.get(metric, 0.0)
    score = max(0.0, 100.0 - (penalty / total_tasks * 100.0))
    score = round(score, 1)

    # Project health
    health = "Excellent ✅" if score > 90 else ("Good 🟡" if score > 75 else "Needs Work 🔴")
    summary_df.loc[len(summary_df)] = ["Schedule Integrity Score (0–100)", score]
    summary_df.loc[len(summary_df)] = ["Project Health", health]

    # Prepare output dict of detail tables
    out = {
        "summary": summary_df,
        "invalid_rows": pd.DataFrame(invalid_rows),
        "cycles": cycles_df,
        "dangling": dangling_df,
        "leadlag": pd.DataFrame(lead_lag_warnings),
        "critical_path_issues": cp_gaps_df,
        "dep_type_counts": pd.DataFrame(list(dep_type_counts.items()), columns=["Dependency_Type", "Count"]),
        "slack": slack_df,
        "constraints": cons_df,
        "no_res": no_res_df,
        "oos": oos_df,
        "no_base": no_base_df,
        "unrealistic": unreal_df,
        "iso_ms": iso_ms_df,
        "bad_names": lq_names_df,
        "baseline_late": base_var_df,
    }
    return out


def write_report(audit: Dict[str, pd.DataFrame], path: str) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        audit["summary"].to_excel(w, sheet_name="Summary", index=False)

        sheet_map = {
            "invalid_rows": "Malformed_Missing",
            "cycles": "Circular_Dependencies",
            "dangling": "Dangling_Tasks",
            "leadlag": "Lead_Lag_Warnings",
            "critical_path_issues": "Critical_Path_Issues",
            "dep_type_counts": "Dependency_Types",
            "slack": "Slack_Flags",
            "constraints": "Constraints",
            "no_res": "No_Resources",
            "oos": "OOS_Actuals",
            "no_base": "No_Baseline",
            "unrealistic": "Unrealistic_Duration",
            "iso_ms": "Isolated_Milestones",
            "bad_names": "Low_Quality_Names",
            "baseline_late": "Baseline_Variance_Late",
        }
        for key, sheet in sheet_map.items():
            df = audit.get(key, pd.DataFrame())
            if df is None:
                df = pd.DataFrame()
            df.to_excel(w, sheet_name=sheet[:31], index=False)


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="GAO Schedule Quality Auditor – Full", layout="wide", page_icon="📘")

st.title("GAO Schedule Quality Auditor — Full Version")
st.caption("Complete GAO-style metrics with robust column detection and configurable thresholds.")

with st.sidebar:
    st.header("Configuration")
    leadlag = st.number_input("Lead/Lag absolute limit (units in text like 10d/4w)", min_value=0, max_value=120, value=10, step=1)
    excessive_slack_days = st.number_input("Excessive Slack threshold (days)", min_value=0, max_value=3650, value=60, step=5)
    long_dur_days = st.number_input("Unrealistic Long Duration (days)", min_value=1, max_value=3650, value=60, step=5)
    short_dur_min = st.number_input("Unrealistic Short Duration (minutes)", min_value=0, max_value=10000, value=60, step=5)
    exclude_summaries = st.checkbox("Exclude summary tasks from most checks", value=True)

st.subheader("📤 Upload your Microsoft Project Excel export (.xlsx)")
uploaded = st.file_uploader("Drag & drop or browse", type=["xlsx"])

if uploaded:
    try:
        # If multiple sheets exist, let user pick (optional)
        xls = pd.ExcelFile(uploaded)
        sheet = st.selectbox("Select worksheet", xls.sheet_names, index=0)
        df = pd.read_excel(uploaded, sheet_name=sheet)
        st.success(f"Loaded: {sheet} ({len(df):,} rows)")

        # Normalize then audit
        df = normalize_df(df)
        audit = run_audit_dataframe(
            df,
            leadlag_abs_limit=leadlag,
            excessive_slack_days=excessive_slack_days,
            long_duration_days=long_dur_days,
            short_duration_minutes=short_dur_min,
            exclude_summaries=exclude_summaries,
        )

        st.markdown("### ✅ Analysis complete!")
        st.dataframe(audit["summary"], use_container_width=True)

        # Download full Excel report
        out_buf = io.BytesIO()
        write_report(audit, path=out_buf)  # type: ignore: ExcelWriter accepts buffer
        st.download_button(
            "⬇️ Download Full Audit Report (Excel)",
            data=out_buf.getvalue(),
            file_name="GAO_Schedule_Audit_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"Processing error: {e}")
        st.exception(e)
else:
    st.info("Upload a .xlsx export from Microsoft Project to begin.")
