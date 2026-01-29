import streamlit as st
import pandas as pd
import os
from datetime import date

ATTENDANCE_FILE = "attendance.csv"
STUDENT_MASTER = "student_master.csv"

st.set_page_config(
    page_title="School Attendance Dashboard",
    layout="wide"
)

st.title("📊 Live Attendance Dashboard")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_attendance():
    if not os.path.exists(ATTENDANCE_FILE):
        return pd.DataFrame()
    return pd.read_csv(ATTENDANCE_FILE)

@st.cache_data
def load_students():
    if not os.path.exists(STUDENT_MASTER):
        return pd.DataFrame()
    return pd.read_csv(STUDENT_MASTER)

attendance = load_attendance()
students = load_students()

if attendance.empty or students.empty:
    st.warning("⚠ No attendance data found yet.")
    st.stop()

attendance["date"] = pd.to_datetime(attendance["date"]).dt.date

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("🔍 Filters")

selected_date = st.sidebar.date_input(
    "Select Date", value=date.today()
)

selected_class = st.sidebar.selectbox(
    "Select Class",
    ["All"] + sorted(students["class"].unique().tolist())
)

selected_section = st.sidebar.selectbox(
    "Select Section",
    ["All"] + sorted(students["section"].unique().tolist())
)

selected_student = st.sidebar.selectbox(
    "Select Student",
    ["All"] + sorted(students["name"].unique().tolist())
)

# -----------------------------
# Apply Filters
# -----------------------------
filtered = attendance[attendance["date"] == selected_date]

if selected_class != "All":
    filtered = filtered[filtered["class"] == selected_class]

if selected_section != "All":
    filtered = filtered[filtered["section"] == selected_section]

if selected_student != "All":
    filtered = filtered[filtered["name"] == selected_student]

# -----------------------------
# KPIs
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("👨‍🎓 Total Students", students.shape[0])
col2.metric("✅ Present", filtered.shape[0])
col3.metric("❌ Absent", students.shape[0] - filtered.shape[0])

# -----------------------------
# Attendance Table
# -----------------------------
st.subheader("📋 Attendance Records")

st.dataframe(
    filtered.sort_values("time"),
    use_container_width=True
)

st.caption("🔄 Dashboard auto-updates when file changes")
