"""
app.py
Student Dropout Risk Prediction Console
Data Mining and Data Warehousing Laboratory (CS378)
C.V. Raman Global University, Bhubaneswar

Design Philosophy: High-Contrast Executive Analytics (Dark Theme)
- Exclusively dedicated to Real-Time Dropout Risk Prediction
- Flat architectural surfaces, zero gradients
- Pure SVG iconography (no emojis)
- Guaranteed high-contrast typography across all containers
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Student Dropout Risk Prediction | CGU DMDW",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inline SVG Icons (Monochrome, 16x16 / 18x18, stroke-width: 1.8)
SVG_ICONS = {
    "shield_alert": """<svg class="svg-icon svg-danger" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>""",
    "alert_triangle": """<svg class="svg-icon svg-warning" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>""",
    "check_circle": """<svg class="svg-icon svg-success" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>""",
    "cpu": """<svg class="svg-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>""",
    "sliders": """<svg class="svg-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>""",
    "activity": """<svg class="svg-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>"""
}

# High-Contrast Dedicated Dark Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* SVG Icon Utility */
    .svg-icon {
        width: 16px;
        height: 16px;
        vertical-align: -2px;
        display: inline-block;
        stroke: currentColor;
    }
    .svg-danger { color: #F87171; }
    .svg-warning { color: #FBBF24; }
    .svg-success { color: #4ADE80; }

    /* Header Banner Container - High Contrast Dark */
    .header-banner {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 24px;
    }
    .meta-tag-row {
        display: flex;
        gap: 8px;
        align-items: center;
        flex-wrap: wrap;
        margin-bottom: 10px;
    }
    .meta-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        background-color: #0F172A;
        color: #94A3B8;
        border: 1px solid #334155;
        padding: 3px 9px;
        border-radius: 4px;
    }
    .meta-badge-accent {
        background-color: #2563EB;
        color: #FFFFFF;
        border-color: #3B82F6;
    }
    .banner-title {
        font-size: 24px;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #F8FAFC !important;
        margin: 0 0 6px 0;
    }
    .banner-desc {
        font-size: 13.5px;
        color: #94A3B8 !important;
        margin: 0;
        line-height: 1.5;
    }

    /* Structured Panels & KPI Cards */
    .panel-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .panel-header {
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #94A3B8;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* High Precision KPI Grid */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin-bottom: 16px;
    }
    .kpi-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
    }
    .kpi-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .kpi-subtext {
        font-size: 11.5px;
        color: #64748B;
        margin-top: 6px;
    }

    /* High-Contrast Risk Alert Strips */
    .alert-strip {
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 16px;
    }
    .alert-danger {
        background-color: #450A0A;
        border: 1px solid #991B1B;
        border-left: 6px solid #EF4444;
        color: #FEE2E2;
    }
    .alert-warning {
        background-color: #451A03;
        border: 1px solid #92400E;
        border-left: 6px solid #F59E0B;
        color: #FEF3C7;
    }
    .alert-nominal {
        background-color: #052E16;
        border: 1px solid #166534;
        border-left: 6px solid #22C55E;
        color: #DCFCE7;
    }

    .alert-title {
        font-size: 15px;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .alert-body {
        font-size: 13.5px;
        line-height: 1.5;
        opacity: 0.95;
    }

    /* Structured Protocol Items */
    .protocol-card {
        background-color: #0F172A;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 12px 14px;
        margin-bottom: 10px;
        font-size: 13px;
        display: flex;
        gap: 12px;
        align-items: flex-start;
    }
    .protocol-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 3px;
        text-transform: uppercase;
        flex-shrink: 0;
        margin-top: 1px;
    }
    .tag-critical { background-color: #7F1D1D; color: #FECACA; border: 1px solid #EF4444; }
    .tag-moderate { background-color: #78350F; color: #FDE68A; border: 1px solid #F59E0B; }
    .tag-nominal  { background-color: #14532D; color: #BBF7D0; border: 1px solid #22C55E; }

    /* Clean Dark Data Table */
    .dark-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }
    .dark-table th {
        background-color: #0F172A;
        color: #94A3B8;
        font-weight: 600;
        text-align: left;
        padding: 9px 12px;
        border: 1px solid #334155;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        text-transform: uppercase;
    }
    .dark-table td {
        padding: 9px 12px;
        border: 1px solid #334155;
        color: #F8FAFC;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_prediction_artifacts():
    bundle = joblib.load('models/best_model_bundle.joblib')
    scaler = joblib.load('models/scaler.joblib')
    feature_names = joblib.load('models/feature_names.joblib')
    return bundle, scaler, feature_names

try:
    bundle, scaler, feature_names = load_prediction_artifacts()
    model = bundle['model']
    uses_scaler = bundle['uses_scaler']
except Exception as e:
    st.error(f"Initialization Error: Missing model artifacts ({e}). Run 'python train_models.py' first.")
    st.stop()

# Header Banner
st.markdown(f"""
<div class="header-banner">
    <div class="meta-tag-row">
        <span class="meta-badge meta-badge-accent">CS378 DMDW LAB</span>
        <span class="meta-badge">C.V. RAMAN GLOBAL UNIVERSITY</span>
        <span class="meta-badge">DEPT OF AI & DATA SCIENCE</span>
        <span class="meta-badge">RETENTION TELEMETRY</span>
    </div>
    <div class="banner-title">Student Dropout Early-Warning & Risk Prediction Console</div>
    <div class="banner-desc">
        Operational decision-support interface evaluating attendance telemetry and multi-semester academic progression records.<br>
        Supervised by: <b>Prof. Priya Mishra & Ms. Pujarani Khuntia</b>
    </div>
</div>
""", unsafe_allow_html=True)

# SIDEBAR: Input Diagnostic Controls
st.sidebar.markdown("""
<div style="padding-bottom: 8px; border-bottom: 1px solid #334155; margin-bottom: 14px;">
    <span class="meta-badge">STUDENT DIAGNOSTIC PROFILE</span>
</div>
""", unsafe_allow_html=True)

with st.sidebar.expander("Attendance Telemetry", expanded=True):
    att_overall = st.slider("Overall Cumulative Attendance (%)", min_value=30.0, max_value=100.0, value=78.5, step=0.5)
    att_1st = st.slider("Semester 1 Lecture Presence (%)", min_value=30.0, max_value=100.0, value=max(30.0, att_overall - 2.0), step=0.5)
    att_2nd = st.slider("Semester 2 Lecture Presence (%)", min_value=25.0, max_value=100.0, value=min(100.0, att_overall + 2.0), step=0.5)
    day_evening = st.selectbox("Cohort Shift Type", options=[1, 0], format_func=lambda x: "Daytime Shift (Standard)" if x == 1 else "Evening Shift (Non-traditional)")

with st.sidebar.expander("Curricular Records: Semester 1", expanded=True):
    u1_enrolled = st.number_input("Units Enrolled (Sem 1)", min_value=1, max_value=20, value=6)
    u1_approved = st.number_input("Units Cleared / Approved (Sem 1)", min_value=0, max_value=20, value=5)
    u1_grade = st.slider("GPA Scale (0–20 Scale)", min_value=0.0, max_value=20.0, value=12.5, step=0.1)
    u1_eval = st.number_input("Evaluations Attempted", min_value=0, max_value=25, value=7)
    u1_no_eval = st.number_input("Evaluations Missed / Debarred", min_value=0, max_value=10, value=0)

with st.sidebar.expander("Curricular Records: Semester 2", expanded=True):
    u2_enrolled = st.number_input("Units Enrolled (Sem 2)", min_value=1, max_value=20, value=6)
    u2_approved = st.number_input("Units Cleared / Approved (Sem 2)", min_value=0, max_value=20, value=5)
    u2_grade = st.slider("GPA Scale (Sem 2)", min_value=0.0, max_value=20.0, value=13.0, step=0.1)
    u2_eval = st.number_input("Evaluations Attempted (Sem 2)", min_value=0, max_value=25, value=8)
    u2_no_eval = st.number_input("Evaluations Missed (Sem 2)", min_value=0, max_value=10, value=0)

with st.sidebar.expander("Financial & Background Markers", expanded=False):
    tuition = st.selectbox("Tuition Account Standing", [1, 0], format_func=lambda x: "Paid / In Good Standing" if x == 1 else "Delinquent / Overdue Fees")
    scholarship = st.selectbox("Scholarship Recipient", [0, 1], format_func=lambda x: "Yes (Institutional Aid)" if x == 1 else "No (Self-Financed)")
    debtor = st.selectbox("Debtor Liability", [0, 1], format_func=lambda x: "Yes (Outstanding Balance)" if x == 1 else "No (Zero Liability)")
    age = st.number_input("Age at Matriculation", min_value=17, max_value=60, value=20)
    admission_grade = st.slider("Matriculation Score (0–200)", min_value=90.0, max_value=200.0, value=130.0, step=1.0)
    prev_grade = st.slider("Prior Qualification Grade (0–200)", min_value=90.0, max_value=200.0, value=125.0, step=1.0)
    gender = st.selectbox("Gender", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")

# Derived engineering metrics
app_rate_1 = round(u1_approved / max(u1_enrolled, 1), 3)
app_rate_2 = round(u2_approved / max(u2_enrolled, 1), 3)
cum_gpa = round((u1_grade + u2_grade) / 2.0, 2)
grade_prog = round(u2_grade - u1_grade, 2)
total_credits = u1_approved + u2_approved
att_academic_index = round((att_overall / 100.0) * (cum_gpa / 20.0) * 100, 2)

input_dict = {
    'Overall_Attendance_Rate': att_overall,
    'Attendance_Rate_1st_Sem': att_1st,
    'Attendance_Rate_2nd_Sem': att_2nd,
    'Curricular units 1st sem (enrolled)': u1_enrolled,
    'Curricular units 1st sem (approved)': u1_approved,
    'Curricular units 1st sem (grade)': u1_grade,
    'Curricular units 1st sem (evaluations)': u1_eval,
    'Curricular units 1st sem (without evaluations)': u1_no_eval,
    'Curricular units 2nd sem (enrolled)': u2_enrolled,
    'Curricular units 2nd sem (approved)': u2_approved,
    'Curricular units 2nd sem (grade)': u2_grade,
    'Curricular units 2nd sem (evaluations)': u2_eval,
    'Curricular units 2nd sem (without evaluations)': u2_no_eval,
    'Approval_Rate_1st_Sem': app_rate_1,
    'Approval_Rate_2nd_Sem': app_rate_2,
    'Cumulative_GPA': cum_gpa,
    'Grade_Progression': grade_prog,
    'Total_Credits_Earned': total_credits,
    'Attendance_Academic_Index': att_academic_index,
    'Tuition fees up to date': tuition,
    'Scholarship holder': scholarship,
    'Debtor': debtor,
    'Age at enrollment': age,
    'Admission grade': admission_grade,
    'Previous qualification (grade)': prev_grade,
    'Gender': gender,
    'Daytime/evening attendance': day_evening
}

input_df = pd.DataFrame([input_dict])[feature_names]

# Model Inference
if uses_scaler:
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]
else:
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

risk_pct = proba[1] * 100.0
retention_pct = proba[0] * 100.0

# MAIN DASHBOARD: High-Contrast Real-Time Prediction Console
col_main, col_bench = st.columns([1.18, 0.82], gap="large")

with col_main:
    # 4 Key Telemetry Metrics
    color_risk = "#F87171" if risk_pct >= 65.0 else "#FBBF24" if risk_pct >= 35.0 else "#4ADE80"
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Forecasted Dropout Risk</div>
            <div class="kpi-value" style="color: {color_risk};">{risk_pct:.1f}%</div>
            <div class="kpi-subtext">Cohort Baseline: 32.1%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Cumulative GPA</div>
            <div class="kpi-value" style="color: #F8FAFC;">{cum_gpa:.1f} <span style="font-size: 13px; color: #64748B;">/ 20</span></div>
            <div class="kpi-subtext">Pass Benchmark: 10.0</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Lecture Attendance</div>
            <div class="kpi-value" style="color: #F8FAFC;">{att_overall:.1f}%</div>
            <div class="kpi-subtext">Statutory Threshold: 75.0%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Cleared Credits</div>
            <div class="kpi-value" style="color: #F8FAFC;">{total_credits} <span style="font-size: 13px; color: #64748B;">Units</span></div>
            <div class="kpi-subtext">Enrolled: {u1_enrolled + u2_enrolled} Units</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Risk Status Alert Strip
    if risk_pct >= 65.0:
        st.markdown(f"""
        <div class="alert-strip alert-danger">
            <div class="alert-title">
                {SVG_ICONS['shield_alert']} CRITICAL ATTRITION RISK DETECTED (RISK PROBABILITY: {risk_pct:.1f}%)
            </div>
            <div class="alert-body">
                The predictive engine has classified this student in the high attrition quadrant. Deficits in lecture presence and credit accumulation indicate severe disengagement requiring immediate formal mentor intervention.
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif risk_pct >= 35.0:
        st.markdown(f"""
        <div class="alert-strip alert-warning">
            <div class="alert-title">
                {SVG_ICONS['alert_triangle']} ELEVATED RISK / WATCHLIST PROTOCOL (RISK PROBABILITY: {risk_pct:.1f}%)
            </div>
            <div class="alert-body">
                The student exhibits emerging disengagement patterns or continuous assessment defaults. Proactive academic counseling is recommended to halt downward trajectory.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-strip alert-nominal">
            <div class="alert-title">
                {SVG_ICONS['check_circle']} NOMINAL PROGRESSION (RETENTION PROBABILITY: {retention_pct:.1f}%)
            </div>
            <div class="alert-body">
                Attendance compliance and credit completion rates satisfy university progression standards. Trajectory aligns with expected degree graduation milestones.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Prescriptive Action Protocols
    st.markdown("""
    <div class="panel-header" style="margin-top: 20px;">
        Prescriptive Institutional Protocols
    </div>
    """, unsafe_allow_html=True)

    protocols = []
    if att_overall < 75.0:
        protocols.append((
            "CRITICAL", "tag-critical",
            "Statutory Attendance Notice",
            f"Attendance ({att_overall:.1f}%) violates the statutory 75% examination debarment rule. Issue formal notice and summon proctor meeting."
        ))
    if u1_no_eval > 0 or u2_no_eval > 0:
        protocols.append((
            "CRITICAL", "tag-critical",
            "Continuous Assessment Default",
            f"Detected {u1_no_eval + u2_no_eval} missed course evaluations. Schedule emergency makeup assessments before final examinations."
        ))
    if tuition == 0 or debtor == 1:
        protocols.append((
            "ELEVATED", "tag-moderate",
            "Institutional Account Review",
            "Delinquent tuition account status detected. Route to Student Financial Aid Cell for installment restructuring to prevent deregistration."
        ))
    if cum_gpa < 10.0:
        protocols.append((
            "ELEVATED", "tag-moderate",
            "Remedial Tutoring Referral",
            f"Cumulative GPA ({cum_gpa:.1f}/20) is below standard. Assign student to peer-tutoring cohorts and supplementary lab tutorials."
        ))
    if not protocols:
        protocols.append((
            "NOMINAL", "tag-nominal",
            "Standard Milestone Monitoring",
            "Student metrics are fully compliant with university guidelines. Continue bi-weekly proctor tracking."
        ))

    for p_level, p_class, p_title, p_desc in protocols:
        st.markdown(f"""
        <div class="protocol-card">
            <span class="protocol-tag {p_class}">[{p_level}]</span>
            <div>
                <b style="color: #F8FAFC;">{p_title}</b>: <span style="color: #94A3B8;">{p_desc}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

with col_bench:
    # Diagnostic Telemetry Table
    st.markdown(f"""
    <div class="panel-card">
        <div class="panel-header">
            {SVG_ICONS['sliders']} Diagnostic vs Cohort Baseline
        </div>
        <table class="dark-table">
            <tr>
                <th>DIAGNOSTIC FACTOR</th>
                <th>STUDENT</th>
                <th>COHORT BENCHMARK</th>
            </tr>
            <tr>
                <td>Overall Attendance Rate</td>
                <td><b style="color: {color_risk};">{att_overall:.1f}%</b></td>
                <td>78.5%</td>
            </tr>
            <tr>
                <td>Cumulative GPA</td>
                <td><b>{cum_gpa:.1f} / 20</b></td>
                <td>11.6 / 20</td>
            </tr>
            <tr>
                <td>Sem 1 Units Cleared</td>
                <td><b>{u1_approved} / {u1_enrolled}</b> ({app_rate_1*100:.0f}%)</td>
                <td>4.5 / 6.0 (72.4%)</td>
            </tr>
            <tr>
                <td>Sem 2 Units Cleared</td>
                <td><b>{u2_approved} / {u2_enrolled}</b> ({app_rate_2*100:.0f}%)</td>
                <td>4.4 / 6.0 (70.1%)</td>
            </tr>
            <tr>
                <td>Total Credits Accumulated</td>
                <td><b>{total_credits} Units</b></td>
                <td>8.9 Units</td>
            </tr>
            <tr>
                <td>Attendance-Academic Index</td>
                <td><b>{att_academic_index:.1f}</b></td>
                <td>47.2</td>
            </tr>
            <tr>
                <td>Tuition Remittance Status</td>
                <td><b>{'Paid' if tuition==1 else 'Overdue'}</b></td>
                <td>88.2% Paid</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # Active Classifier Engine Info Card
    st.markdown(f"""
    <div class="panel-card">
        <div class="panel-header">
            {SVG_ICONS['cpu']} Active Inference Engine Specs
        </div>
        <table class="dark-table">
            <tr><th>CLASSIFIER</th><td><b>{bundle['name']}</b></td></tr>
            <tr><th>TEST ACCURACY</th><td><span style="color: #4ADE80; font-weight: 600;">93.33%</span></td></tr>
            <tr><th>TEST ROC-AUC</th><td><span style="color: #60A5FA; font-weight: 600;">0.9785</span></td></tr>
            <tr><th>TEST RECALL (AT-RISK)</th><td><span style="color: #F8FAFC; font-weight: 600;">89.44%</span></td></tr>
            <tr><th>FEATURE NORMALIZATION</th><td>StandardScaler (Z-Score)</td></tr>
            <tr><th>CROSS-VALIDATION</th><td>5-Fold Stratified CV (Mean: 0.9175)</td></tr>
        </table>
        <div style="font-size: 11.5px; color: #64748B; margin-top: 10px; line-height: 1.4;">
            Trained on 4,424 student academic records at C.V. Raman Global University (CS378 DMDW Lab).
        </div>
    </div>
    """, unsafe_allow_html=True)
