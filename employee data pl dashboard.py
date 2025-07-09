import streamlit as st
import pandas as pd
import plotly.express as px
import yagmail
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import logging
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import re
from fuzzywuzzy import fuzz, process
import os
import numpy as np
import time
import json
import io

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration with a custom background
st.set_page_config(page_title="Employee Performance Dashboard", layout="wide")
st.markdown(
    """
    <style>
    .reportview-container {
        background: #E6F2FF;
    }
    .sidebar .sidebar-content {
        background: #2E2E2E;
        color: white;
    }
    .chat-message {
        padding: 10px;
        margin: 5px;
        border-radius: 10px;
        background-color: #f1f1f1;
    }
    .user-message {
        background-color: #1E90FF;
        color: white;
        text-align: right;
    }
    .bot-message {
        background-color: #696969;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Email Setup ---
sender_email = "akhilmiriyala998@gmail.com"
receiver_admin_email = "akhilmiriyala998@gmail.com"
email_password = st.secrets.get("email_password", None)
if email_password:
    yag = yagmail.SMTP(user=sender_email, password=email_password)
else:
    yag = None
    logging.warning("Email password not found in secrets. Email functionality disabled.")

# Title
st.title("Employee Performance Dashboard")
st.markdown("Interactive visualizations of employee performance metrics.")

# --- Chatbot Data Loading and Processing ---
CACHE_FILE = 'sheet_cache.pkl'
CACHE_EXPIRY = 3600  # seconds (1 hour)

def is_cache_expired():
    return not os.path.exists(CACHE_FILE) or (time.time() - os.path.getmtime(CACHE_FILE)) > CACHE_EXPIRY

COLUMN_ALIASES = {
    'sallary': 'Salary',
    'perfomance': 'Performance Score',
    'satisfcation': 'Satisfaction Score',
    'dept': 'Department',
    'employeeid': 'Employee ID',
    'id': 'Employee ID'
}

def load_google_sheet(sheet_url, retries=3):
    """Load data from a Google Sheet."""
    start_time = time.time()
    if not is_cache_expired():
        logging.info("Loading cached data...")
        with open(CACHE_FILE, 'rb') as f:
            df = pd.read_pickle(f)
        logging.info(f"Loaded cached data in {time.time() - start_time:.2f} seconds")
        return df
    for attempt in range(retries):
        try:
            scope = ['https://www.googleapis.com/auth/spreadsheets']
            # Load credentials from Streamlit secrets
            creds_json = st.secrets.get("google_sheets", {}).get("credentials", None)
            if not creds_json:
                logging.error("Google Sheets credentials not found in secrets. Data loading failed.")
                return None
            creds_dict = json.loads(creds_json)
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
            client = gspread.authorize(creds)
            sheet = client.open_by_url(sheet_url).sheet1
            data = sheet.get_all_records()
            df = pd.DataFrame(data)
            df = df.dropna(axis=1, how='all')
            with open(CACHE_FILE, 'wb') as f:
                pd.to_pickle(df, f)
            logging.info(f"Loaded and cached Google Sheet in {time.time() - start_time:.2f} seconds")
            return df
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                logging.warning(f"Rate limit hit, retrying in {2 ** attempt} seconds...")
                time.sleep(2 ** attempt)
            else:
                logging.error(f"Error loading Google Sheet: {e}")
                return None
    return None

def assign_performance_level(scores):
    scores = pd.to_numeric(scores, errors='coerce')
    return np.select([scores > 75, scores >= 50, scores < 50], ['High', 'Medium', 'Low'], default='Unknown')

def assign_satisfaction_level(scores):
    scores = pd.to_numeric(scores, errors='coerce')
    return np.select([scores > 4, scores >= 3, scores < 3], ['High', 'Medium', 'Low'], default='Unknown')

def assign_retention_risk(perf_scores, sat_scores):
    perf_scores = pd.to_numeric(perf_scores, errors='coerce')
    sat_scores = pd.to_numeric(sat_scores, errors='coerce')
    return np.select([(perf_scores > 75) & (sat_scores > 4), (perf_scores < 50) | (sat_scores < 3)], ['Low', 'High'], default='Medium')

def find_best_column(question, columns):
    question = question.lower()
    for alias, actual in COLUMN_ALIASES.items():
        if alias in question and actual in columns:
            return actual, 100
    best_match, score = process.extractOne(question, columns, scorer=fuzz.token_sort_ratio)
    return best_match, score if score >= 50 else 0

def generate_visualization(result, intent, group_by=None, agg_col=None, agg_type=None):
    try:
        if intent == 'aggregate' and not result.empty:
            if agg_type == 'count' and group_by:
                fig = px.pie(result, names=group_by, values='Count', title=f'Employee Count by {group_by.capitalize()}')
                return fig
            elif agg_type in ['mean', 'sum'] and group_by and agg_col:
                fig = px.bar(result, x=group_by, y=agg_col, title=f'{agg_type.capitalize()} {agg_col.capitalize()} by {group_by.capitalize()}')
                return fig
        return None
    except Exception as e:
        logging.error(f"Error generating visualization: {e}")
        return None

def process_question(question, df):
    start_time = time.time()
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "Error: No data loaded from Google Sheet. Please ensure the sheet is accessible."

    question = question.lower().strip()
    columns = df.columns.str.lower().tolist()

    if 'columns' in question and 'google sheet' in question:
        result = pd.DataFrame({'Columns': df.columns.tolist()})
        logging.info(f"Processed column query in {time.time() - start_time:.2f} seconds")
        return result.to_markdown(index=False)

    keywords = []
    fuzzy_matches = []
    for word in question.split():
        matched_col, score = find_best_column(word, columns)
        if matched_col and score >= 50:
            keywords.append(matched_col)
            if score < 90:
                fuzzy_matches.append(f"Interpreting '{word}' as '{matched_col}' (similarity: {score}%)")

    if not keywords and 'columns' not in question:
        return f"No matching columns found. Available columns: {', '.join(df.columns)}"

    emp_id_match = re.search(r'\b\d+\b', question)
    emp_id = int(emp_id_match.group()) if emp_id_match else None

    visualize = any(keyword in question for keyword in ['show', 'plot', 'graph', 'chart'])

    if any(word in question for word in ['who', 'name']):
        intent = 'retrieve_name'
    elif any(word in question for word in ['average', 'sum', 'total', 'count']):
        intent = 'aggregate'
    else:
        intent = 'retrieve'

    try:
        if intent == 'retrieve_name':
            if emp_id:
                result = df[df['Employee ID'] == emp_id][['Name']]
            else:
                keyword = next((k for k in keywords if k != 'name'), 'department')
                search_term = question.split()[-1]
                result = df[df[keyword].str.contains(search_term, case=False, na=False)][['Name']]

        elif intent == 'aggregate':
            agg_type = 'mean' if 'average' in question else 'sum' if 'sum' in question else 'count'
            group_by = next((col for col in keywords if col not in df.select_dtypes(include=['int64', 'float64']).columns), None)
            if group_by:
                if agg_type == 'count':
                    result = df.groupby(group_by).size().reset_index(name='Count')
                else:
                    agg_col = next((col for col in keywords if col in df.select_dtypes(include=['int64', 'float64']).columns), None)
                    if agg_col:
                        result = df.groupby(group_by)[agg_col].agg(agg_type).reset_index()
                    else:
                        result = pd.DataFrame()
            else:
                agg_col = next((col for col in keywords if col in df.select_dtypes(include=['int64', 'float64']).columns), None)
                if agg_col:
                    result = pd.DataFrame({agg_col: [df[agg_col].agg(agg_type)]})
                else:
                    result = pd.DataFrame()

        else:
            if emp_id:
                result = df[df['Employee ID'] == emp_id][keywords]
            else:
                keyword = keywords[0] if keywords else 'department'
                search_term = question.split()[-1]
                result = df[df[keyword].str.contains(search_term, case=False, na=False)][keywords]

        visual_message = None
        if visualize and intent in ['aggregate'] and not result.empty:
            fig = generate_visualization(result, intent, group_by, agg_col if intent == 'aggregate' else None, agg_type if intent == 'aggregate' else None)
            visual_message = "Visualization generated (use fig.show() in Jupyter or save manually)" if fig else "Failed to generate visualization"

        table = result.to_markdown(index=False) if not result.empty else "No data found for the query."

        response = table
        if fuzzy_matches:
            response = f"{' '.join(fuzzy_matches)}\n\n" + response
        if visual_message:
            response += f"\n\nVisualization: {visual_message}"

        logging.info(f"Processed query in {time.time() - start_time:.2f} seconds")
        return response

    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return f"Error processing query: {e}\nAvailable columns: {', '.join(df.columns)}"

# Load data
sheet_url = "https://docs.google.com/spreadsheets/d/1OxU_4C8zAp_3sqcmj2dnn4YB7N6xcI6PUPLWSG-yl4E/edit?usp=sharing"
df = load_google_sheet(sheet_url) if is_cache_expired() else pd.read_pickle(CACHE_FILE)

if df is None or df.empty:
    st.error("Failed to load data. Using empty DataFrame.")
    logger.error("Failed to load data. Using empty DataFrame.")
    df = pd.DataFrame()

# Preprocessing
if not df.empty:
    try:
        df['Hire_Date'] = pd.to_datetime(df['Hire_Date'], errors='coerce')
        df['Years_At_Company'] = (pd.Timestamp.now() - df['Hire_Date']).dt.days / 365.25
        if 'Performance Score' in df.columns:
            df['Performance_Level'] = assign_performance_level(df['Performance Score'])
        if 'Satisfaction Score' in df.columns:
            df['Satisfaction_Level'] = assign_satisfaction_level(df['Satisfaction Score'])
        if 'Performance Score' in df.columns and 'Satisfaction Score' in df.columns:
            df['Retention_Risk_Level'] = assign_retention_risk(df['Performance Score'], df['Satisfaction Score'])
        if 'Annual Salary' in df.columns:
            df['Annual Salary'] = df['Annual Salary'].replace('[\$,]', '', regex=True).astype(float)
            logger.info(f"Annual Salary data type after conversion: {df['Annual Salary'].dtype}")
        else:
            logger.warning("Annual Salary column not found in the dataset.")
        logger.info("Preprocessing completed successfully.")
    except Exception as e:
        st.error(f"Error during preprocessing: {str(e)}")
        logger.error(f"Preprocessing error: {str(e)}")

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar: Filters and Chatbot
st.sidebar.header("Filters")
departments = df['Department'].dropna().unique().tolist() if not df.empty else []
job_titles = df['Job_Title'].dropna().unique().tolist() if not df.empty else []
remote_options = ['All', 'Work From Home', 'Work From Office', 'Hybrid']
employee_ids = ['All'] + df['Employee_ID'].dropna().astype(str).unique().tolist() if not df.empty else ['All']

selected_employee = st.sidebar.selectbox("Select Employee ID", employee_ids)
selected_department = st.sidebar.selectbox("Select Department", ["All"] + departments)
selected_job = st.sidebar.selectbox("Select Job Title", ["All"] + job_titles)
selected_remote = st.sidebar.selectbox("Select Remote Work Type", remote_options)
date_range = st.sidebar.date_input("Filter by Hire Date Range", [df['Hire_Date'].min(), df['Hire_Date'].max()] if not df.empty else [datetime.now(), datetime.now()])

# Apply filters
filtered_df = df.copy() if not df.empty else pd.DataFrame()
if not filtered_df.empty:
    try:
        if selected_employee != "All":
            filtered_df = filtered_df[filtered_df['Employee_ID'] == selected_employee]
        if selected_department != "All":
            filtered_df = filtered_df[filtered_df['Department'] == selected_department]
        if selected_job != "All":
            filtered_df = filtered_df[filtered_df['Job_Title'] == selected_job]
        if selected_remote != "All":
            filtered_df = filtered_df[filtered_df['Remote_Work_Category'] == selected_remote]
        if len(date_range) == 2:
            filtered_df = filtered_df[(filtered_df['Hire_Date'] >= pd.to_datetime(date_range[0])) &
                                      (filtered_df['Hire_Date'] <= pd.to_datetime(date_range[1]))]
        logger.info(f"Filtered data to {len(filtered_df)} rows.")
    except Exception as e:
        st.error(f"Error applying filters: {str(e)}")
        logger.error(f"Filter error: {str(e)}")

# Sidebar: Chatbot
st.sidebar.header("Employee Data Chatbot")
st.sidebar.markdown("Ask about employee details, aggregations, or columns (e.g., 'employee 123 performance score', 'average salary by department').")
st.sidebar.markdown("Use keywords like 'plot' or 'chart' for visualizations.")
user_input = st.sidebar.text_input("Type your question:", key="chat_input")
if user_input:
    st.session_state.chat_history.append({"role": "user", "message": user_input})
    response = process_question(user_input, df)
    st.session_state.chat_history.append({"role": "bot", "message": response})
    st.sidebar.markdown(f"<div class='chat-message bot-message'>{response}</div>", unsafe_allow_html=True)

# Display chat history
for chat in st.session_state.chat_history:
    if chat['role'] == 'user':
        st.sidebar.markdown(f"<div class='chat-message user-message'>{chat['message']}</div>", unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f"<div class='chat-message bot-message'>{chat['message']}</div>", unsafe_allow_html=True)

# Email Alerts
st.subheader("Email Notifications")
if st.button("Send Email Alerts"):
    if yag is None:
        st.error("Email functionality disabled due to missing password in secrets.")
    else:
        low_sat_df = df[df['Satisfaction_Level'] == 'Low'] if not df.empty else pd.DataFrame()
        low_sat_content = "\n".join([f"Low Satisfaction - EmpID: {row['Employee_ID']}, Dept: {row['Department']}, Job: {row['Job_Title']}" for _, row in low_sat_df.iterrows()]) if not low_sat_df.empty else "None"
        low_perf_df = df[df['Performance_Level'] == 'Low'] if not df.empty else pd.DataFrame()
        low_perf_content = "\n".join([f"Low Performance - EmpID: {row['Employee_ID']}, Dept: {row['Department']}, Job: {row['Job_Title']}" for _, row in low_perf_df.iterrows()]) if not low_perf_df.empty else "None"
        high_ret_df = df[df['Retention_Risk_Level'] == 'High'] if not df.empty else pd.DataFrame()
        high_ret_content = "\n".join([f"High Retention Risk - EmpID: {row['Employee_ID']}, Dept: {row['Department']}, Job: {row['Job_Title']}" for _, row in high_ret_df.iterrows()]) if not high_ret_df.empty else "None"
        
        email_content = f"Employee Alerts:\n\nLow Satisfaction Alerts:\n{low_sat_content}\n\nLow Performance Alerts:\n{low_perf_content}\n\nHigh Retention Risk Alerts:\n{high_ret_content}"
        if low_sat_content != "None" or low_perf_content != "None" or high_ret_content != "None":
            try:
                yag.send(to=receiver_admin_email, subject="🚨 Employee Alerts", contents=email_content)
                st.success("✅ Admin alert email sent with low satisfaction, low performance, and high retention risk details.")
                logger.info("Email alert sent successfully.")
            except Exception as e:
                st.error(f"Failed to send email: {str(e)}")
                logger.error(f"Email sending error: {str(e)}")
        else:
            st.info("ℹ️ No email alerts sent. No employees meet alert criteria.")
else:
    st.info("ℹ️ Click 'Send Email Alerts' to send notifications manually.")

# Alert Display Buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("View Low Satisfaction Alerts"):
        st.write("**Low Satisfaction Alerts**")
        st.dataframe(low_sat_df[['Employee_ID', 'Department', 'Job_Title']] if 'low_sat_df' in locals() and not low_sat_df.empty else pd.DataFrame())
with col2:
    if st.button("View Low Performance Alerts"):
        st.write("**Low Performance Alerts**")
        st.dataframe(low_perf_df[['Employee_ID', 'Department', 'Job_Title']] if 'low_perf_df' in locals() and not low_perf_df.empty else pd.DataFrame())
with col3:
    if st.button("View High Retention Risk Alerts"):
        st.write("**High Retention Risk Alerts**")
        st.dataframe(high_ret_df[['Employee_ID', 'Department', 'Job_Title']] if 'high_ret_df' in locals() and not high_ret_df.empty else pd.DataFrame())

# KPI Cards
remote_efficiency_column = None
for col in df.columns:
    if col.lower().replace(" ", "_") == 'remote_work_efficiency':
        remote_efficiency_column = col
        break

if not df.empty:
    if remote_efficiency_column:
        remote_work_efficiency_avg = filtered_df[remote_efficiency_column].mean() if not filtered_df.empty else 0
    else:
        st.warning("Remote Work Efficiency column not found in the Google Sheet.")
        remote_work_efficiency_avg = 0

    productivity_avg = filtered_df['Productivity score'].mean() if not filtered_df.empty else 0
    avg_salary = filtered_df['Annual Salary'].mean() if 'Annual Salary' in filtered_df.columns and not filtered_df.empty else 0
    if avg_salary == 0 and 'Annual Salary' in filtered_df.columns:
        st.warning("Average Salary is 0. Check if Annual Salary contains valid numeric data.")
    num_employees = len(filtered_df)
else:
    remote_work_efficiency_avg = 0
    productivity_avg = 0
    avg_salary = 0
    num_employees = 0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
        <div style="background-color:black; padding:20px; border-radius:10px">
            <h3 style="color:white; text-align:center;">Remote Work Efficiency</h3>
            <h1 style="color:white; text-align:center;">{remote_work_efficiency_avg:.2f}</h1>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
        <div style="background-color:black; padding:20px; border-radius:10px">
            <h3 style="color:white; text-align:center;">Productivity Score</h3>
            <h1 style="color:white; text-align:center;">{productivity_avg:.2f}</h1>
        </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
        <div style="background-color:black; padding:20px; border-radius:10px">
            <h3 style="color:white; text-align:center;">Average Annual Salary</h3>
            <h1 style="color:white; text-align:center;">${avg_salary:,.2f}</h1>
        </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
        <div style="background-color:black; padding:20px; border-radius:10px">
            <h3 style="color:white; text-align:center;">Number of Employees</h3>
            <h1 style="color:white; text-align:center;">{num_employees}</h1>
        </div>
    """, unsafe_allow_html=True)

# Visual Analytics
st.subheader("Visual Analytics")
row1_col1, row1_col2, row1_col3 = st.columns(3)
with row1_col1:
    st.markdown("**Remote Work Efficiency by Department**")
    remote_efficiency = filtered_df.groupby(['Department', 'Remote_Work_Category'])['Productivity score'].mean().reset_index() if not filtered_df.empty else pd.DataFrame()
    fig_remote = px.bar(remote_efficiency, x='Department', y='Productivity score', color='Remote_Work_Category',
                        barmode='group', color_discrete_map={'Work From Home': '#1E90FF', 'Work From Office': '#696969', 'Hybrid': '#228B22'}) if not remote_efficiency.empty else px.bar()
    st.plotly_chart(fig_remote, use_container_width=True)
with row1_col2:
    st.markdown("**Performance Level Distribution by Job Title**")
    tree_data = filtered_df.groupby(['Job_Title', 'Performance_Level'])['Employee_ID'].count().reset_index() if not filtered_df.empty else pd.DataFrame()
    tree_data.rename(columns={'Employee_ID': 'Number_of_Employees'}, inplace=True)
    fig_tree = px.treemap(tree_data, path=['Job_Title', 'Performance_Level'], values='Number_of_Employees',
                          color='Performance_Level',
                          color_discrete_map={'Low': '#FF4040', 'Medium': '#FFA500', 'High': '#228B22'}) if not tree_data.empty else px.treemap()
    st.plotly_chart(fig_tree, use_container_width=True)
with row1_col3:
    st.markdown("**Employee Count by Retention Risk Level and Job Title**")
    retention_count = filtered_df.groupby(['Job_Title', 'Retention_Risk_Level'])['Employee_ID'].count().reset_index() if not filtered_df.empty else pd.DataFrame()
    retention_count.rename(columns={'Employee_ID': 'Number_of_Employees'}, inplace=True)
    fig_ret = px.bar(retention_count, x='Job_Title', y='Number_of_Employees', color='Retention_Risk_Level',
                     color_discrete_map={'Low': '#8B0000', 'Medium': '#FFA500', 'High': '#006400'}) if not retention_count.empty else px.bar()
    st.plotly_chart(fig_ret, use_container_width=True)

row2_col1, row2_col2, row2_col3 = st.columns(3)
with row2_col1:
    st.markdown("**Remote Work Type Distribution**")
    remote_data = filtered_df['Remote_Work_Category'].value_counts().reset_index() if not filtered_df.empty else pd.DataFrame()
    remote_data.columns = ['Remote_Work_Category', 'Count']
    fig_pie = px.pie(remote_data, names='Remote_Work_Category', values='Count',
                     color_discrete_map={'Work From Home': '#1E90FF', 'Work From Office': '#696969', 'Hybrid': '#228B22'}) if not remote_data.empty else px.pie()
    st.plotly_chart(fig_pie, use_container_width=True)
with row2_col2:
    st.markdown("**Average Satisfaction by Department**")
    sat_avg = filtered_df.groupby('Department')['Employee_Satisfaction_Score'].mean().reset_index() if not filtered_df.empty else pd.DataFrame()
    fig_sat = px.bar(sat_avg, x='Department', y='Employee_Satisfaction_Score',
                     color='Department', color_discrete_sequence=px.colors.qualitative.Plotly) if not sat_avg.empty else px.bar()
    st.plotly_chart(fig_sat, use_container_width=True)
with row2_col3:
    st.markdown("**Performance Trend by Years at Company**")
    if not filtered_df.empty and 'Years_At_Company' in filtered_df.columns:
        filtered_df['Years_At_Company'] = pd.to_numeric(filtered_df['Years_At_Company'], errors='coerce')
        years_bins = pd.cut(filtered_df['Years_At_Company'], bins=10, include_lowest=True)
        filtered_df['Years_Bin'] = years_bins.apply(lambda x: x.mid if pd.notna(x) and hasattr(x, 'mid') else np.nan)
        trend_data = filtered_df.groupby(['Years_Bin', 'Job_Title'], dropna=False)['Performance_Score'].mean().reset_index()
    else:
        filtered_df['Years_Bin'] = pd.Series()
        trend_data = pd.DataFrame()
    fig_line = px.line(trend_data, x='Years_Bin', y='Performance_Score', color='Job_Title') if not trend_data.empty else px.line()
    st.plotly_chart(fig_line, use_container_width=True)

# Data Alert Tables
st.subheader("Data Alerts")
alert_dept = st.selectbox("Filter Alerts by Department", ["All"] + departments, key="alert_dept")
alert_job = st.selectbox("Filter Alerts by Job Title", ["All"] + job_titles, key="alert_job")

# Apply alert filters
low_sat_alert_df = df[df['Satisfaction_Level'] == 'Low'][['Employee_ID', 'Department', 'Job_Title']] if not df.empty else pd.DataFrame()
high_ret_alert_df = df[df['Retention_Risk_Level'] == 'High'][['Employee_ID', 'Department', 'Job_Title']] if not df.empty else pd.DataFrame()

if alert_dept != "All":
    low_sat_alert_df = low_sat_alert_df[low_sat_alert_df['Department'] == alert_dept] if not low_sat_alert_df.empty else pd.DataFrame()
    high_ret_alert_df = high_ret_alert_df[high_ret_alert_df['Department'] == alert_dept] if not high_ret_alert_df.empty else pd.DataFrame()
if alert_job != "All":
    low_sat_alert_df = low_sat_alert_df[low_sat_alert_df['Job_Title'] == alert_job] if not low_sat_alert_df.empty else pd.DataFrame()
    high_ret_alert_df = high_ret_alert_df[high_ret_alert_df['Job_Title'] == alert_job] if not high_ret_alert_df.empty else pd.DataFrame()

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Low Satisfaction Alerts**")
    st.dataframe(low_sat_alert_df, use_container_width=True)
with col2:
    st.markdown("**High Retention Risk Alerts**")
    st.dataframe(high_ret_alert_df, use_container_width=True)

# Auto-refresh
st_autorefresh(interval=300000, key="refresh")