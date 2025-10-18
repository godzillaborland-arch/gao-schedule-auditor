# GAO Schedule Auditor (Streamlit App)

Upload your Microsoft Project schedule export (**.xlsx**) to audit against GAO Schedule Assessment Guide best practices.

## Features
- Detect malformed/missing links, circulars, dangling tasks
- Lead/lag warnings, FS logic gaps, slack checks
- Constraints (hard-date) recognition
- Baseline variance (Finish vs Baseline Finish) with Top-10 delays
- Overall **Schedule Health Score (0–100)** with traffic-light status
- Downloadable Excel report with all tabs

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
