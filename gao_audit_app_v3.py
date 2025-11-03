import pandas as pd
import numpy as np

# --- Utility: Clean and normalize Total Slack column ---
def clean_slack_column(df):
    if "Total Slack" not in df.columns:
        return pd.Series(dtype=float)

    slack = df["Total Slack"].astype(str)
    slack = (
        slack.str.replace("days", "", regex=False)
              .str.replace("d", "", regex=False)
              .str.replace("?", "", regex=False)
              .str.replace("−", "-", regex=False)   # handles unicode minus
              .str.strip()
    )

    # Convert to numeric, coercing errors to NaN
    slack = pd.to_numeric(slack, errors="coerce").fillna(0)
    return slack


# --- Main audit function ---
def run_audit(df):
    results = []

    # --- 1. Slack Metrics ---
    slack = clean_slack_column(df)
    results.append({"Metric": "Negative Slack", "Value": int((slack < 0).sum())})
    results.append({"Metric": "Excessive Slack (>60d)", "Value": int((slack > 60).sum())})

    # --- 2. Logic Links ---
    pred_col = next((c for c in df.columns if "Predecessor" in c), None)
    succ_col = next((c for c in df.columns if "Successor" in c), None)

    missing_preds = df[pred_col].isna().sum() if pred_col else 0
    missing_succs = df[succ_col].isna().sum() if succ_col else 0
    results.append({"Metric": "Missing Predecessors", "Value": missing_preds})
    results.append({"Metric": "Missing Successors", "Value": missing_succs})

    # --- 3. Constraints ---
    const_col = next((c for c in df.columns if "Constraint" in c), None)
    if const_col:
        valid_constraints = ["As Soon As Possible", "None", "NA", "", np.nan]
        const_count = df[~df[const_col].isin(valid_constraints)].shape[0]
    else:
        const_count = 0
    results.append({"Metric": "Constraints", "Value": const_count})

    # --- 4. Summary ---
    total_tasks = len(df)
    score = max(0, 100 - (missing_preds + missing_succs + const_count + (slack < 0).sum()) / total_tasks * 100)
    results.append({"Metric": "Schedule Integrity Score (0–100)", "Value": round(score, 1)})

    health = (
        "Excellent ✅" if score >= 90
        else "Good 🟢" if score >= 75
        else "Needs Work 🔴"
    )
    results.append({"Metric": "Project Health", "Value": health})

    return pd.DataFrame(results)

