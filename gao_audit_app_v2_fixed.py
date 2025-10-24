import streamlit as st
import pandas as pd
from io import BytesIO

# --- GAO Audit Core ---
def run_audit(df):
    # Normalize column names to lowercase and remove spaces/underscores
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # Handle multiple variants of percent complete
    if 'percent_complete' in df.columns:
        df['percent_complete_std'] = df['percent_complete']
    elif 'percentcomplete' in df.columns:
        df['percent_complete_std'] = df['percentcomplete']
    else:
        df['percent_complete_std'] = 0

    # Ensure Predecessors column exists
    if 'predecessors' not in df.columns:
        df['predecessors'] = ''

    # Compute simple metrics
    malformed_links = df['predecessors'].isna().sum()
    incomplete_tasks = (df['percent_complete_std'] < 1).sum()

    summary = pd.DataFrame({
        'Metric': ['Malformed/Missing Links', 'Incomplete Tasks'],
        'Value': [malformed_links, incomplete_tasks]
    })

    return summary

# --- Streamlit UI ---
st.set_page_config(page_title='GAO Schedule Quality Auditor', layout='wide', page_icon='📘')
st.title('GAO Schedule Quality Auditor – Fixed Version')
st.caption('Now compatible with both Percent Complete and Percent_Complete columns.')

uploaded_file = st.file_uploader('📤 Upload your Microsoft Project Excel (.xlsx)', type=['xlsx'])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    with st.spinner('🔍 Analyzing schedule... please wait'):
        results_df = run_audit(df)

    st.success('✅ Analysis complete!')
    st.dataframe(results_df)

    # Export results to Excel
    output = BytesIO()
    results_df.to_excel(output, index=False)
    st.download_button(
        label='⬇️ Download Full Audit Report',
        data=output.getvalue(),
        file_name='GAO_Schedule_Audit_Report_Fixed.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
else:
    st.info('Please upload a valid Microsoft Project Excel export (.xlsx) to begin.')

st.markdown('---')
st.caption('© 2025 GAO Schedule Quality Auditor | Built with ❤️ using Streamlit')
