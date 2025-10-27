# gao_core.py
# Shared core for GAO schedule validation, autofix, and auditing.
# - Accepts either a pandas DataFrame or an Excel path / file-like object
# - Performs column aliasing, duration autofix (“1638 days” -> 1638), validation summary
# - Runs GAO checks and returns (summary_df, detail_dict)
from __future__ import annotations
import re
from typing import Tuple, Dict, Any, List, Union
import pandas as pd
import numpy as np
from collections import Counter
import networkx as nx

# -------- Config (defaults; can be overridden by callers) ----------
DEFAULTS = {
    "LEAD_LAG_ABS_LIMIT": 10,        # flag offsets > 10 units (+/-)
    "LONG_DURATION_DAYS": 60,        # > 60 days
    "SHORT_DURATION_MIN": 60,        # < 60 minutes
    "EXCESSIVE_SLACK_DAYS": 60,      # excessive total slack threshold
    "IGNORE_CONSTRAINTS": {"As Soon As Possible"},  # treated as benign
}

# Column aliases: we’ll map these to canonical names
COLUMN_ALIASES = {
    "UID": ["UID", "Unique ID", "Unique_ID", "UniqueID", "Task UID", "ID"],
    "Task Name": ["Task Name", "Name", "Activity Name", "Task"],
    "Start": ["Start", "Start_Date", "Start Date", "Begin"],
    "Finish": ["Finish", "Finish_Date", "Finish Date", "End"],
    "Duration": ["Duration", "Duration (minutes)", "Duration_Minutes"],
    "Predecessors": ["Predecessors", "Pred", "Predecessor", "Predecessor(s)"],
    "Successors": ["Successors", "Successor", "Successor(s)"],
    "Percent Complete": ["Percent Complete", "Percent_Complete", "% Complete", "Pct Complete"],
    "Baseline Finish": ["Baseline Finish", "Baseline_Finish", "BL Finish", "BL_Finish"],
    "Total Slack": ["Total Slack", "Total_Slack", "Slack", "TotalSlack"],
    "Summary": ["Summary"],
    "Milestone": ["Milestone", "Is Milestone", "Is_Milestone"],
    "Resource Names": ["Resource Names", "ResourceNames", "Resources"],
    "Constraint Type": ["Constraint Type", "Constraint_Type", "ConstraintType"],
    "Actual Start": ["Actual Start", "Actual_Start"],
}

REQUIRED_FOR_VALIDATION = [
    "Task Name", "Start", "Finish", "Duration", "Predecessors", "Percent Complete"
]

GAO_WEIGHTS = {
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

# ------------- Helpers -------------
def _first_present(df: pd.DataFrame, candidates: List[str]) -> str | None:
    colmap = {c.lower().strip(): c for c in df.columns}
    for c in candidates:
        key = c.lower().strip()
        if key in colmap:
            return colmap[key]
    return None

def _apply_aliases(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    issues: List[str] = []
    rename_map: Dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        found = _first_present(df, aliases)
        if found and found != canonical:
            rename_map[found] = canonical
    if rename_map:
        df = df.rename(columns=rename_map)
    for col in REQUIRED_FOR_VALIDATION:
        if col not in df.columns:
            issues.append(f"❌ Missing required column: '{col}'")
    if rename_map:
        for old, new in rename_map.items():
            issues.append(f"ℹ Column renamed automatically: '{old}' → '{new}'")
    return df, issues, rename_map

def _clean_duration_series(s: pd.Series) -> Tuple[pd.Series, int]:
    """Silently convert textual durations to numeric days; 'NA','—' -> 0."""
    fixed = 0
    out = []
    for v in s.fillna(""):
        if isinstance(v, (int, float, np.number)):
            out.append(float(v)); continue
        text = str(v).strip().lower()
        if text in {"", "na", "n/a", "none", "-", "—"}:
            out.append(0.0); 
            if text: fixed += 1
            continue
        m = re.search(r"([+-]?\d+(\.\d+)?)", text)
        if m:
            num = float(m.group(1))
            if "week" in text:  # rough scale
                num *= 7.0
            if m.group(0) != text:
                fixed += 1
            out.append(num)
        else:
            out.append(0.0); fixed += 1
    return pd.Series(out, index=s.index, dtype=float), fixed

def _parse_dependency_tokens(text: str) -> Tuple[List[tuple], List[str]]:
    """Parse strings like '101FS, 102SS+3d' into (pred_id, type, offset)."""
    tokens, invalid = [], []
    if not isinstance(text, str) or not text.strip():
        return tokens, invalid
    for raw in re.split(r"[,\s]+", text.strip()):
        if not raw: continue
        m = re.match(r"^(\d+)([FS]{1,2})?([+-]?\d+[hdwmy])?$", raw, re.I)
        if m:
            tokens.append((int(m.group(1)), (m.group(2) or "FS").upper(), m.group(3)))
        else:
            invalid.append(raw)
    return tokens, invalid

def _minutes_to_days(minutes) -> float | None:
    try:
        return float(minutes) / 480.0  # 8h/day
    except Exception:
        return None

def _duration_flags(duration_minutes: float, long_days: float, short_min: float) -> List[str]:
    flags = []
    d_days = _minutes_to_days(duration_minutes)
    if d_days is None:
        return flags
    if d_days > long_days:
        flags.append("Unrealistic Duration: Long")
    elif duration_minutes < short_min:
        flags.append("Unrealistic Duration: Short")
    return flags

# ------------- Validation + Autofix -------------
def validate_and_autofix(
    data: Union[str, "pathlib.Path", Any, pd.DataFrame],
    sheet_name: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load (if needed), alias columns, clean durations, build validation summary.
    Returns: (df_clean, validation_summary_df, messages)
    """
    # Load
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        if sheet_name:
            df = pd.read_excel(data, sheet_name=sheet_name)
        else:
            df = pd.read_excel(data)

    df, msgs, _rename = _apply_aliases(df)

    counts = {
        "Missing Required Columns": sum(1 for c in REQUIRED_FOR_VALIDATION if c not in df.columns),
        "Blank Task Names": int(df["Task Name"].isna().sum()) if "Task Name" in df.columns else 0,
        "Invalid Start/Finish Dates": 0,
        "Duration Auto-Fixed": 0,
        "Empty Baseline Finish": int(df["Baseline Finish"].isna().sum()) if "Baseline Finish" in df.columns else 0,
        "Blank Predecessors": int(df["Predecessors"].astype(str).str.strip().eq("").sum()) if "Predecessors" in df.columns else 0,
        "Blank Successors": int(df["Successors"].astype(str).str.strip().eq("").sum()) if "Successors" in df.columns else 0,
    }

    # Dates
    for dc in ("Start", "Finish"):
        if dc in df.columns:
            dt = pd.to_datetime(df[dc], errors="coerce")
            counts["Invalid Start/Finish Dates"] += int(dt.isna().sum())

    # Duration
    if "Duration" in df.columns:
        df["Duration"], fixed = _clean_duration_series(df["Duration"].astype(object))
        counts["Duration Auto-Fixed"] = int(fixed)
    else:
        msgs.append("❌ Missing 'Duration' after alias mapping.")

    summary = pd.DataFrame({"Check": list(counts.keys()), "Detected Issues": list(counts.values())})
    return df, summary, msgs

# ------------- GAO Audit -------------
def run_gao_audit(
    df: pd.DataFrame,
    *,
    excessive_slack_days: int = DEFAULTS["EXCESSIVE_SLACK_DAYS"],
    long_duration_days: int = DEFAULTS["LONG_DURATION_DAYS"],
    short_duration_min: int = DEFAULTS["SHORT_DURATION_MIN"],
    lead_lag_abs_limit: int = DEFAULTS["LEAD_LAG_ABS_LIMIT"],
    benign_constraints: set[str] = DEFAULTS["IGNORE_CONSTRAINTS"],
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Run GAO checks and return (summary_df, detail_dfs).
    detail_dfs contains multiple DataFrames keyed by sheet-like names.
    """
    # Normalize basic fields
    for col in ["Predecessors", "Successors", "Resource Names"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    if "UID" not in df.columns:
        df["UID"] = range(1, len(df) + 1)
    if "Milestone" not in df.columns:
        df["Milestone"] = False
    if "Summary" not in df.columns:
        df["Summary"] = False

    # 1) Parse links, invalids, and build graph
    valid_links: List[tuple] = []
    invalid_rows: List[dict] = []
    lead_lag_warnings: List[dict] = []

    for _, row in df.iterrows():
        preds, invalid = _parse_dependency_tokens(row["Predecessors"])
        if invalid:
            invalid_rows.append({"UID": row["UID"], "Task Name": row.get("Task Name", row.get("Name", "(unnamed)")),
                                 "Malformed_Tokens": ", ".join(invalid)})
        for pred_id, dep_type, offset in preds:
            if offset:
                try:
                    n = int(re.findall(r"[+-]?\d+", offset)[0])
                    if abs(n) > lead_lag_abs_limit:
                        lead_lag_warnings.append({"UID": row["UID"], "Task Name": row.get("Task Name", "(unnamed)"),
                                                  "LeadLag": offset})
                except Exception:
                    pass
            valid_links.append((pred_id, row["UID"], dep_type))

    G = nx.DiGraph()
    G.add_edges_from([(a, b) for a, b, _ in valid_links])
    try:
        cycles = list(nx.find_cycle(G, orientation="original"))
    except nx.NetworkXNoCycle:
        cycles = []

    dep_type_counts = Counter([t for _, _, t in valid_links])
    dep_types_df = pd.DataFrame(list(dep_type_counts.items()), columns=["Dependency_Type", "Count"])

    # 2) Dangling + critical-path logic gaps
    all_ids = set(df["UID"].tolist())
    connected = {a for a, _, _ in valid_links} | {b for _, b, _ in valid_links}
    dangling_ids = all_ids - connected
    dangling_df = df[df["UID"].isin(dangling_ids)][["UID", "Task Name"]].copy()

    incoming_fs = {b for a, b, t in valid_links if t == "FS"}
    outgoing_fs = {a for a, b, t in valid_links if t == "FS"}
    no_fs = all_ids - (incoming_fs | outgoing_fs)
    cp_issues = df[df["UID"].isin(no_fs)][["UID", "Task Name"]].copy()

    # 3) Slack (assuming Total Slack in minutes; convert to days if numbers look large)
    neg_slack = pd.DataFrame(columns=["UID", "Task Name", "Total Slack (days)"])
    big_slack = pd.DataFrame(columns=["UID", "Task Name", "Total Slack (days)"])
    if "Total Slack" in df.columns:
        ts = pd.to_numeric(df["Total Slack"], errors="coerce")
        # If many values > 1000, assume minutes and convert to days
        assumed_days = ts.copy()
        if (ts > 1000).sum() > 0:
            assumed_days = ts / 480.0
        neg_slack = df[assumed_days < 0][["UID", "Task Name"]].assign(**{"Total Slack (days)": assumed_days[assumed_days < 0]})
        big_slack = df[assumed_days > excessive_slack_days][["UID", "Task Name"]].assign(**{"Total Slack (days)": assumed_days[assumed_days > excessive_slack_days]})

    # 4) Constraints (ignore benign)
    constraints = pd.DataFrame(columns=["UID", "Task Name", "Constraint Type"])
    if "Constraint Type" in df.columns:
        s = df["Constraint Type"].astype(str).str.strip()
        mask = (~s.eq("")) & (~s.isin(benign_constraints))
        constraints = df[mask][["UID", "Task Name", "Constraint Type"]]

    # 5) Resources missing (non-summary)
    no_res = df[(~df["Summary"].astype(bool)) & (df["Resource Names"].str.strip() == "")][["UID", "Task Name"]]

    # 6) Out-of-sequence actuals (simple)
    oos = pd.DataFrame(columns=["UID", "Task Name", "Detail"])
    if "Actual Start" in df.columns and df["Actual Start"].notna().any():
        for _, row in df.iterrows():
            if pd.isna(row.get("Actual Start")):
                continue
            preds, _ = _parse_dependency_tokens(row["Predecessors"])
            for pred_id, _, _ in preds:
                pred = df[df["UID"] == pred_id]
                if not pred.empty:
                    pf = pred.iloc[0].get("Finish")
                    if pd.notna(pf) and pd.notna(row["Actual Start"]) and pd.to_datetime(row["Actual Start"]) < pd.to_datetime(pf):
                        oos.loc[len(oos)] = [row["UID"], row["Task Name"], f"Started before pred {pred_id} finished"]
                        break

    # 7) Baseline variance (late)
    baseline_var = pd.DataFrame(columns=["UID", "Task Name", "Baseline_Finish", "Finish", "Late_Days"])
    if "Baseline Finish" in df.columns and "Finish" in df.columns:
        bf = pd.to_datetime(df["Baseline Finish"], errors="coerce")
        fn = pd.to_datetime(df["Finish"], errors="coerce")
        late_mask = (bf.notna() & fn.notna() & (fn > bf))
        baseline_var = df[late_mask][["UID", "Task Name"]].copy()
        baseline_var["Baseline_Finish"] = bf[late_mask].values
        baseline_var["Finish"] = fn[late_mask].values
        baseline_var["Late_Days"] = (fn[late_mask] - bf[late_mask]).dt.days

    # 8) Unrealistic durations (if Duration stored as minutes in source, caller should pass minutes; here we assume days already)
    unrealistic = pd.DataFrame(columns=["UID", "Task Name", "Reason"])
    if "Duration" in df.columns:
        # If Duration looks like minutes (many > 1000), treat as minutes for short threshold
        dur = pd.to_numeric(df["Duration"], errors="coerce")
        if (dur > 1000).sum() > 0:
            # minutes-based short duration
            short_mask = dur < short_duration_min
            long_mask = (dur / 480.0) > long_duration_days
        else:
            short_mask = (dur * 480.0) < short_duration_min
            long_mask = dur > long_duration_days
        if short_mask.any():
            unrealistic = pd.concat([unrealistic,
                                     df[short_mask][["UID", "Task Name"]].assign(Reason="Unrealistic Duration: Short")])
        if long_mask.any():
            unrealistic = pd.concat([unrealistic,
                                     df[long_mask][["UID", "Task Name"]].assign(Reason="Unrealistic Duration: Long")])

    # 9) Isolated milestones & low-quality names
    iso_ms = df[(df["Milestone"].astype(bool)) &
                ((df["Predecessors"].str.strip() == "") | (df["Successors"].str.strip() == ""))][["UID", "Task Name"]]
    bad_names = df[(~df["Summary"].astype(bool)) & (df["Task Name"].astype(str).str.len() < 10)][["UID","Task Name"]]

    # Build summary metrics
    summary_items = {
        "Malformed/Missing Links": len(invalid_rows),
        "Circular Dependencies": len(cycles),
        "Dangling Tasks": len(dangling_df),
        "Lead/Lag Warnings": len(lead_lag_warnings),
        "Critical Path Logic Gaps": len(cp_issues),
        "Negative Slack": len(neg_slack),
        "Excessive Slack": len(big_slack),
        "Constraints": len(constraints),
        "No Resources": len(no_res),
        "Out-of-Sequence Actuals": len(oos),
        "No Baseline": int(df["Baseline Finish"].isna().sum()) if "Baseline Finish" in df.columns else 0,
        "Unrealistic Duration": len(unrealistic),
        "Isolated Milestone": len(iso_ms),
        "Low-Quality Name": len(bad_names),
        "Baseline Variance (Late)": len(baseline_var),
    }
    summary_df = pd.DataFrame(list(summary_items.items()), columns=["Metric", "Value"])

    # Compute score
    total_tasks = max(1, len(df))
    penalty = 0.0
    for metric, val in summary_items.items():
        if isinstance(val, (int, float)):
            penalty += float(val) * GAO_WEIGHTS.get(metric, 0.0)
    score = max(0.0, 100.0 - (penalty / total_tasks * 100.0))
    score = round(score, 1)
    health = "Excellent ✅" if score > 90 else "Good 🟡" if score > 75 else "Needs Work 🔴"
    summary_df.loc[len(summary_df)] = ["Schedule Integrity Score (0–100)", score]
    summary_df.loc[len(summary_df)] = ["Project Health", health]

    # Detail dict
    details = {
        "Malformed_Missing": pd.DataFrame(invalid_rows),
        "Circular_Dependencies": pd.DataFrame(cycles, columns=["From", "To", "Type"]) if cycles else pd.DataFrame(columns=["From","To","Type"]),
        "Dependency_Types": dep_types_df,
        "Dangling_Tasks": dangling_df,
        "Lead_Lag_Warnings": pd.DataFrame(lead_lag_warnings),
        "Critical_Path_Issues": cp_issues,
        "Negative_Slack": neg_slack,
        "Excessive_Slack": big_slack,
        "Constraints": constraints,
        "No_Resources": no_res,
        "OOS_Actuals": oos,
        "Baseline_Late": baseline_var,
        "Unrealistic_Duration": unrealistic,
        "Isolated_Milestones": iso_ms,
        "Low_Quality_Names": bad_names,
    }
    return summary_df, details
