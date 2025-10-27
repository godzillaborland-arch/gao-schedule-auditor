# gao_full_audit_cli.py
# Command-line GAO audit. Produces a multi-sheet Excel report.
import sys
import pandas as pd
from gao_core import validate_and_autofix, run_gao_audit, DEFAULTS

OUTPUT_XLSX = "GAO_Schedule_Audit_Report.xlsx"

def write_report(summary_df, details, output_path=OUTPUT_XLSX):
    with pd.ExcelWriter(output_path, engine="openpyxl") as w:
        summary_df.to_excel(w, sheet_name="Summary", index=False)
        for name, df in details.items():
            df.to_excel(w, sheet_name=name[:31], index=False)

def main():
    if len(sys.argv) < 2:
        print("Usage: python gao_full_audit_cli.py <excel_path> [sheet_name]")
        sys.exit(1)
    path = sys.argv[1]
    sheet = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"📄 Loading: {path}")
    df_clean, vsummary, msgs = validate_and_autofix(path, sheet_name=sheet)
    print("\n📊 Validation Summary:")
    print(vsummary.to_string(index=False))
    if msgs:
        print("\nNotes:")
        for m in msgs:
            print("-", m)

    print("\n🔧 Running GAO audit…")
    summary_df, details = run_gao_audit(
        df_clean,
        excessive_slack_days=DEFAULTS["EXCESSIVE_SLACK_DAYS"],
        long_duration_days=DEFAULTS["LONG_DURATION_DAYS"],
        short_duration_min=DEFAULTS["SHORT_DURATION_MIN"],
        lead_lag_abs_limit=DEFAULTS["LEAD_LAG_ABS_LIMIT"],
    )
    write_report(summary_df, details, OUTPUT_XLSX)
    print(f"\n✅ Report saved: {OUTPUT_XLSX}")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
