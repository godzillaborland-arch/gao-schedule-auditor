import pandas as pd

def run_audit(file):
    """
    Universal GAO Schedule Quality Auditor entry point.
    - Accepts either a path to an Excel file or a pandas DataFrame.
    - Returns a DataFrame with audit results (or preview).
    """

    # --- Step 1: Load Data ---
    if isinstance(file, pd.DataFrame):
        df = file.copy()
    else:
        try:
            df = pd.read_excel(file)
        except Exception as e:
            raise ValueError(f"❌ Error loading Excel file: {e}")

    # --- Step 2: Validate Columns ---
    required_cols = ["Task Name", "Start", "Finish", "Duration", "Predecessors"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        msg = f"⚠️ Missing required columns: {', '.join(missing)}"
        print(msg)
        # Create empty results for clarity
        return pd.DataFrame({
            "Metric": ["Missing Columns"],
            "Value": [msg]
        })

    # --- Step 3: Example Audit Checks ---
    results = []

    # Example 1: Missing Predecessors
    missing_preds = df["Predecessors"].isna().sum()
    results.append(("Tasks Missing Predecessors", int(missing_preds)))

    # Example 2: Tasks with Duration > 60 days
    try:
        df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")
        long_tasks = (df["Duration"] > 60).sum()
        results.append(("Unrealistic Duration (>60 days)", int(long_tasks)))
    except Exception:
        results.append(("Unrealistic Duration (>60 days)", "Error parsing durations"))

    # Example 3: Tasks without assigned resources (if column exists)
    if "Resource Names" in df.columns:
        no_res = df["Resource Names"].fillna("").eq("").sum()
        results.append(("Tasks Without Resources", int(no_res)))

    # --- Step 4: Compute Project Health ---
    total_tasks = len(df)
    healthy = total_tasks - (missing_preds + long_tasks)
    score = max(0, min(100, round((healthy / total_tasks) * 100, 1)))

    results.append(("Schedule Integrity Score (0–100)", score))
    results.append(("Project Health", "Excellent ✅" if score > 80 else "Needs Work 🔴"))

    # --- Step 5: Return DataFrame ---
    return pd.DataFrame(results, columns=["Metric", "Value"])


# --- CLI usage ---
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python gao_audit.py <path_to_excel>")
    else:
        path = sys.argv[1]
        print("📄 Loading:", path)
        df_results = run_audit(path)
        print("\n✅ Audit Summary:")
        print(df_results)
