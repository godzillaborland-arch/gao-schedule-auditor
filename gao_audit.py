def run_audit(file):
    import pandas as pd
    import networkx as nx

    def run_audit(file):
    """
    Runs GAO audit on either an uploaded file (.xlsx) or a DataFrame.
    """
    # ✅ Handle both Excel file and already-loaded DataFrame
    if isinstance(file, pd.DataFrame):
        df = file.copy()
    else:
        df = pd.read_excel(file)

    # now continue your audit logic below...
    # e.g., df = normalize_columns(df)
    ...


    # --- Load Excel ---
    # df = pd.read_excel(file)

    # --- Normalize field names ---
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # --- Normalize key fields ---
    for col in ["Predecessors", "Successors", "ResourceNames"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
        else:
            df[col] = ""

    # --- UID detection ---
    possible_uid_columns = ["UID", "Unique_ID", "UniqueID", "Task_UID", "ID"]
    uid_col = next((col for col in df.columns if col.strip() in possible_uid_columns), None)

    if uid_col is None:
        return pd.DataFrame({"Error": ["No UID column found in uploaded file"]})

    # --- Core schedule checks ---
    results = []

    for _, t in df.iterrows():
        issues = []

        # Overdue check
        if "Percent_Complete" in df.columns and "Finish" in df.columns:
            try:
                if float(t.get("Percent_Complete", 0)) < 100 and pd.to_datetime(t["Finish"]) < pd.Timestamp.today():
                    issues.append("Overdue")
            except Exception:
                pass

        # Manual tasks
        if "Manual" in df.columns and t["Manual"] == True:
            issues.append("Manual Task")

        # Missing logic
        if not t["Predecessors"]:
            issues.append("Missing Predecessor")
        if not t["Successors"]:
            issues.append("Missing Successor")

        # Slack
        if "Total_Slack" in df.columns:
            try:
                if t["Total_Slack"] > 50000:
                    issues.append("Excessive Slack")
            except Exception:
                pass

        # Constraint
        if "Constraint_Type" in df.columns and str(t["Constraint_Type"]).lower() not in ["as soon as possible", ""]:
            issues.append(f"Constraint: {t['Constraint_Type']}")

        # Record result
        results.append({
            "Task_Name": t.get("Name", "Unnamed Task"),
            "UID": t.get(uid_col, ""),
            "Issues": ", ".join(issues) if issues else "OK"
        })

    results_df = pd.DataFrame(results)

    # --- Health summary ---
    total = len(results_df)
    ok = (results_df["Issues"] == "OK").sum()
    score = round((ok / total) * 100, 1) if total > 0 else 0

    results_df["Health_Score"] = score

    return results_df
