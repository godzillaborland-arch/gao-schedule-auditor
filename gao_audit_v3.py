#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GAO Schedule Auditor v3 (CLI + importable)
- Robust Total Slack parsing (handles "10 days", "2400 mins", "-3d", numbers)
- Negative & Excessive Slack (threshold in DAYS)
- Missing Predecessors / Successors
- Constraints (GAO-style: non-ASAP flagged)
- Baseline Variance (Late > 5 days) with per-row variance
- Low-Quality Name detection
- Severity column on detail sheets
- Versioned Excel output (…_v5_1.xlsx, …_v5_2.xlsx, …)

CLI:
  python gao_audit_v3.py "CARES V1 Project Schedule.xlsx" --sheet "Task_Table1" --excess-slack-days 60 --out "GAO_Schedule_Audit_Report_v5.xlsx"

Import (Streamlit or other):
  from gao_audit_v3 import run_audit_dataframe, write_excel_report
"""

import argparse
import os
import re
import sys
from datetime import datetime
import numpy as np
import pandas as pd

# ==========================
# Utilities
# ==========================
def find_col(df, patterns, regex=False):
    """
    Return the first matching column name whose name matches any pattern.
    - patterns: list[str]
    - regex: if True, treat patterns as regex substrings (case-insensitive)
    """
    cols = list(df.columns)
    if not regex:
        # exact match
        for p in patterns:
            if p in cols:
                return p
        # normalized (lower, remove spaces)
        norm = {c.lower().replace(" ", ""): c for c in cols}
        for p in patterns:
            key = p.lower().replace(" ", "")
            if key in norm:
                return norm[key]
        return None
    else:
        for c in cols:
            for p in patterns:
                if re.search(p, c, flags=re.IGNORECASE):
                    return c
        return None


_UNIT_TO_DAYS = {
    "min": 1/480.0, "mins": 1/480.0, "minute": 1/480.0, "minutes": 1/480.0,
    "h": 1/8.0, "hr": 1/8.0, "hrs": 1/8.0, "hour": 1/8.0, "hours": 1/8.0,
    "d": 1.0, "day": 1.0, "days": 1.0,
    "w": 7.0, "wk": 7.0, "wks": 7.0, "week": 7.0, "weeks": 7.0,
}
_NUM_UNIT_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*([A-Za-z]*)\s*$")

def series_to_days(s: pd.Series) -> pd.Series:
    """
    Convert a series with numbers or strings like "10 days", "2400 mins", "-3d" into float days.
    - If purely numeric, decides minutes vs days by magnitude (median heuristic).
    """
    if pd.api.types.is_numeric_dtype(s):
        s = pd.to_numeric(s, errors="coerce")
        med = np.nanmedian(np.abs(s)) if s.notna().any() else 0.0
        return (s / 480.0) if med > 1000 else s.astype(float)

    out = []
    for val in s.astype(str).fillna(""):
        v = val.strip()
        if v == "" or v.upper() in {"NA", "N/A", "NONE"}:
            out.append(np.nan)
            continue
        m = _NUM_UNIT_RE.match(v)
        if not m:
            # Fallback: extract any numeric token
            m2 = re.search(r"([+-]?\d+(?:\.\d+)?)", v)
            out.append(float(m2.group(1)) if m2 else np.nan)
            continue
        num = float(m.group(1))
        unit = (m.group(2) or "").lower().strip(".")
        if unit == "":
            out.append(num/480.0 if abs(num) > 1000 else num)
        else:
            out.append(num * _UNIT_TO_DAYS.get(unit, 1.0))
    return pd.Series(out, index=s.index, dtype="float64")


def parse_date_series(s: pd.Series) -> pd.Series:
    """Coerce a date-like column to pandas datetime (NaT on failure)."""
    return pd.to_datetime(s, errors="coerce")


def ensure_versioned_filename(path: str) -> str:
    """
    If 'path' exists, append _1, _2, ... before extension.
    Example: GAO_Schedule_Audit_Report_v5.xlsx -> GAO_Schedule_Audit_Report_v5_1.xlsx
    """
    base, ext = os.path.splitext(path)
    if not os.path.exists(path):
        return path
    i = 1
    while True:
        cand = f"{base}_{i}{ext}"
        if not os.path.exists(cand):
            return cand
        i += 1


# ==========================
# Core audit
# ==========================
def run_audit_dataframe(
    df: pd.DataFrame,
    excess_slack_days: int = 60,
    baseline_late_days: int = 5
):
    """
    Returns (summary_df, detail_dict)
    Sheets in detail_dict (DataFrames):
      - Negative_Slack
      - Excessive_Slack
      - Missing_Preds
      - Missing_Succs
      - Constraints
      - Baseline_Late
      - Low_Quality_Names
    """

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Candidate detail export columns (best-effort)
    candidate_detail_cols = [
        "UID","Unique ID","ID","Task UID","WBS","Outline Number",
        "Name","Task Name","Start","Start Date","Start_Date","Start_Date2",
        "Finish","Finish Date","Finish_Date",
        "Duration","Total Slack","TotalSlack","Slack","Total Float","TotalFloat",
        "Predecessors","Successors","Percent Complete","% Complete","Percent_Complete",
        "Constraint Type","ConstraintType","Task Constraint Type","TaskConstraintType",
        "Constraint Date","ConstraintDate","Task Constraint Date","TaskConstraintDate",
        "Baseline Start","BaselineStart","BL Start","BLStart",
        "Baseline Finish","BaselineFinish","BL Finish","BLFinish"
    ]
    def pick_cols(df, extra=[]):
        cols = []
        for c in candidate_detail_cols + extra:
            if c in df.columns and c not in cols:
                cols.append(c)
        return cols if cols else list(df.columns)

    # ---------- Slack ----------
    slack_col = find_col(
        df, ["Total Slack","TotalSlack","Slack","Total Float","TotalFloat"], regex=False
    ) or find_col(df, [r"slack"], regex=True)

    slack_note = "No Slack column found"
    df["__SlackDays"] = np.nan
    if slack_col:
        df["__SlackDays"] = series_to_days(df[slack_col])
        slack_note = f"Using slack column: {slack_col} (normalized to days)"

    neg_slack_df = pd.DataFrame()
    exc_slack_df = pd.DataFrame()
    neg_count = 0
    exc_count = 0
    if slack_col:
        neg_mask = df["__SlackDays"] < 0
        exc_mask = df["__SlackDays"] > float(excess_slack_days)
        neg_slack_df = df.loc[neg_mask, pick_cols(df, ["__SlackDays"])].copy()
        exc_slack_df = df.loc[exc_mask, pick_cols(df, ["__SlackDays"])].copy()
        # Severity for slack
        if not neg_slack_df.empty:
            neg_slack_df["Severity"] = "Critical"
        if not exc_slack_df.empty:
            exc_slack_df["Severity"] = "Warning"
        neg_count = int(neg_mask.sum())
        exc_count = int(exc_mask.sum())

    # ---------- Predecessors / Successors ----------
    pred_col = find_col(df, [r"predecessor"], regex=True)
    succ_col = find_col(df, [r"successor"], regex=True)

    miss_pred_df = pd.DataFrame()
    miss_succ_df = pd.DataFrame()
    miss_pred_count = "N/A"
    miss_succ_count = "N/A"

    if pred_col:
        preds = df[pred_col].astype(str).fillna("").str.strip()
        miss_pred_mask = preds.eq("")
        miss_pred_df = df.loc[miss_pred_mask, pick_cols(df)].copy()
        if not miss_pred_df.empty:
            miss_pred_df["Severity"] = "Warning"
        miss_pred_count = int(miss_pred_mask.sum())

    if succ_col:
        succs = df[succ_col].astype(str).fillna("").str.strip()
        miss_succ_mask = succs.eq("")
        miss_succ_df = df.loc[miss_succ_mask, pick_cols(df)].copy()
        if not miss_succ_df.empty:
            miss_succ_df["Severity"] = "Warning"
        miss_succ_count = int(miss_succ_mask.sum())

    # ---------- Constraints (GAO: non-ASAP flagged) ----------
    constraint_cols = [c for c in df.columns if re.search(r"constraint", c, re.IGNORECASE)]
    cons_df = pd.DataFrame()
    cons_count_non_asap = 0
    if constraint_cols:
        # Prefer a "type" column if available
        type_like = [c for c in constraint_cols if re.search(r"type", c, re.IGNORECASE)]
        cons_col = type_like[0] if type_like else constraint_cols[0]
        cons_vals = df[cons_col].astype(str).fillna("")
        # Non-ASAP/ALAP only (GAO-style)
        non_asap_mask = cons_vals.str.contains(
            r"Must|No Earlier|No Later|Start On|Finish On|Equal", case=False, na=False
        ) | cons_vals.str.contains(r"MSO|MFO|SNET|FNLT|FNET|SNLT|MSO|MFO", case=False, na=False)
        cons_df = df.loc[non_asap_mask, pick_cols(df, [cons_col])].copy()
        if not cons_df.empty:
            cons_df["Severity"] = "Warning"
        cons_count_non_asap = int(non_asap_mask.sum())

    # ---------- Baseline Variance (Late > 5d) ----------
    finish_col = find_col(df, ["Finish","Finish Date","Finish_Date"], regex=False) or \
                 find_col(df, [r"finish"], regex=True)
    bl_finish_col = find_col(df, ["Baseline Finish","BaselineFinish","BL Finish","BLFinish","Baseline End","BaselineEnd"], regex=False) or \
                    find_col(df, [r"baseline.*finish|baseline.*end"], regex=True)

    baseline_late_df = pd.DataFrame()
    baseline_late_count = 0
    if finish_col and bl_finish_col:
        fin = parse_date_series(df[finish_col])
        blf = parse_date_series(df[bl_finish_col])
        variance_days = (fin - blf).dt.days
        late_mask = variance_days > baseline_late_days
        baseline_late_df = df.loc[late_mask, pick_cols(df, [finish_col, bl_finish_col])].copy()
        if not baseline_late_df.empty:
            baseline_late_df["Variance_Days"] = variance_days[late_mask].values
            # Severity by how late
            baseline_late_df["Severity"] = np.where(
                baseline_late_df["Variance_Days"] > 20, "Critical", "Warning"
            )
        baseline_late_count = int(late_mask.sum())

    # ---------- Low-Quality Names ----------
    name_col = find_col(df, ["Name","Task Name"], regex=False) or find_col(df, [r"name"], regex=True)
    low_name_df = pd.DataFrame()
    low_name_count = 0
    if name_col:
        names = df[name_col].astype(str).fillna("").str.strip()
        too_short = names.str.len() < 3
        bad_tokens = names.str.contains(r"\b(TBD|TEST|DUMMY)\b", case=False, na=False)
        mostly_numeric = names.str.fullmatch(r"\d{3,}", na=False)
        low_mask = too_short | bad_tokens | mostly_numeric
        low_name_df = df.loc[low_mask, pick_cols(df, [name_col])].copy()
        if not low_name_df.empty:
            low_name_df["Severity"] = "Info"
        low_name_count = int(low_mask.sum())

    # ---------- Summary ----------
    summary = pd.DataFrame(
        [
            ["Slack Status", slack_note],
            ["Negative Slack", neg_count],
            [f"Excessive Slack (>{excess_slack_days}d)", exc_count],
            ["Missing Predecessors", miss_pred_count],
            ["Missing Successors", miss_succ_count],
            ["Constraints (GAO-flag, non-ASAP)", cons_count_non_asap],
            [f"Baseline Variance (Late > {baseline_late_days}d)", baseline_late_count],
            ["Low-Quality Name", low_name_count],
        ],
        columns=["Metric", "Value"]
    )

    detail = {
        "Negative_Slack": neg_slack_df,
        "Excessive_Slack": exc_slack_df,
        "Missing_Preds": miss_pred_df,
        "Missing_Succs": miss_succ_df,
        "Constraints": cons_df,
        "Baseline_Late": baseline_late_df,
        "Low_Quality_Names": low_name_df,
    }
    return summary, detail


def write_excel_report(summary_df, detail_dict, out_path="GAO_Schedule_Audit_Report_v5.xlsx"):
    out_path = ensure_versioned_filename(out_path)
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        summary_df.to_excel(w, sheet_name="Summary", index=False)
        for name, df in detail_dict.items():
            sheet = name[:31] if name else "Sheet"
            if df is None or df.empty:
                pd.DataFrame({"Info": [f"No rows for {name}"]}).to_excel(w, sheet_name=sheet, index=False)
            else:
                df.to_excel(w, sheet_name=sheet, index=False)
    return out_path


# ==========================
# CLI
# ==========================
def main():
    p = argparse.ArgumentParser(description="GAO Schedule Audit v3 (plain CLI)")
    p.add_argument("excel_path", help="Path to MS Project Excel export (.xlsx)")
    p.add_argument("--sheet", default=None, help="Optional sheet name to read")
    p.add_argument("--excess-slack-days", type=int, default=60, help="Excessive slack threshold (days)")
    p.add_argument("--baseline-late-days", type=int, default=5, help="Baseline lateness threshold (days)")
    p.add_argument("--out", default="GAO_Schedule_Audit_Report_v5.xlsx", help="Output Excel filename (will be versioned)")
    args = p.parse_args()

    if not os.path.exists(args.excel_path):
        print(f"❌ File not found: {args.excel_path}")
        sys.exit(2)

    try:
        df = pd.read_excel(args.excel_path, sheet_name=args.sheet) if args.sheet else pd.read_excel(args.excel_path)
    except Exception as e:
        print(f"❌ Failed to read Excel: {e}")
        sys.exit(3)

    print(f"📄 Loaded: {args.excel_path} (sheet={args.sheet or 'default'})")
    print("🔧 Running GAO audit…")

    summary_df, detail = run_audit_dataframe(
        df,
        excess_slack_days=args.excess_slack_days,
        baseline_late_days=args.baseline_late_days
    )

    # Plain summary print
    print("\n=== GAO Schedule Audit — Summary ===")
    for _, row in summary_df.iterrows():
        print(f"{row['Metric']}: {row['Value']}")

    out_file = write_excel_report(summary_df, detail, out_path=args.out)
    print(f"\n✅ Report written: {out_file}")


if __name__ == "__main__":
    main()
