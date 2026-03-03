import streamlit as st
import pandas as pd
import joblib

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Prediction | Bhadrak Autonomous College",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">

<style>
/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0e1a !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(99,102,241,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(16,185,129,0.12) 0%, transparent 60%),
        #0a0e1a !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(15,18,32,0.96) !important;
    border-right: 1px solid rgba(99,102,241,0.25) !important;
}
[data-testid="stSidebar"] * { color: #c4c9e2 !important; }

.sidebar-brand {
    text-align: center;
    padding: 1.8rem 1rem 1rem;
    border-bottom: 1px solid rgba(99,102,241,0.2);
    margin-bottom: 1.5rem;
}
.sidebar-brand .brand-icon {
    font-size: 2.8rem;
    display: block;
    margin-bottom: 0.4rem;
    filter: drop-shadow(0 0 12px rgba(99,102,241,0.6));
}
.sidebar-brand h2 {
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    color: #a5b4fc !important;
    line-height: 1.4;
    letter-spacing: 0.03em;
}
.sidebar-brand p {
    font-size: 0.72rem;
    color: #6b7280 !important;
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ── Nav Buttons ── */
/* Hide the "Navigation" label */
[data-testid="stSidebar"] .stRadio > label:first-child {
    display: none !important;
}
/* Hide the raw radio circle inputs */
[data-testid="stSidebar"] .stRadio [data-testid="stWidgetLabel"] { display: none !important; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] { gap: 0.4rem !important; display: flex !important; flex-direction: column !important; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
    background: rgba(99,102,241,0.06) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 10px !important;
    padding: 0.75rem 1rem !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
    transition: all 0.2s ease !important;
    font-size: 0.88rem !important;
    color: #c4c9e2 !important;
    cursor: pointer !important;
    width: 100% !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
    background: rgba(99,102,241,0.18) !important;
    border-color: rgba(99,102,241,0.5) !important;
    color: #a5b4fc !important;
}
/* Hide the actual circle radio input dot */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}
/* Active / selected nav item */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label[data-checked="true"],
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) {
    background: rgba(99,102,241,0.22) !important;
    border-color: rgba(99,102,241,0.6) !important;
    color: #a5b4fc !important;
}

/* ── Main Content ── */
.block-container {
    padding: 2.5rem 3rem !important;
    max-width: 1200px !important;
}

/* ── Page Header ── */
.page-header {
    text-align: center;
    margin-bottom: 3rem;
    padding-bottom: 2rem;
    border-bottom: 1px solid rgba(99,102,241,0.2);
    position: relative;
}
.page-header::after {
    content: '';
    position: absolute;
    bottom: -1px;
    left: 50%;
    transform: translateX(-50%);
    width: 80px;
    height: 2px;
    background: linear-gradient(90deg, #6366f1, #10b981);
}
.page-header .eyebrow {
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #6366f1;
    font-weight: 600;
    margin-bottom: 0.8rem;
}
.page-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2rem, 4vw, 3rem);
    color: #f1f5f9;
    line-height: 1.15;
    letter-spacing: -0.02em;
    margin-bottom: 0.8rem;
}
.page-header h1 span {
    background: linear-gradient(135deg, #818cf8, #10b981);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.page-header p {
    color: #94a3b8;
    font-size: 0.95rem;
    font-weight: 300;
    max-width: 500px;
    margin: 0 auto;
    line-height: 1.7;
}

/* ── Metric Cards ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2.5rem;
}
.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 14px;
    padding: 1.4rem;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent);
}
.metric-card:hover {
    background: rgba(99,102,241,0.07);
    border-color: rgba(99,102,241,0.4);
    transform: translateY(-2px);
}
.metric-card .metric-icon { font-size: 1.6rem; margin-bottom: 0.6rem; }
.metric-card .metric-value {
    font-size: 1.7rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.03em;
}
.metric-card .metric-label {
    font-size: 0.78rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

/* ── Feature Cards (Home) ── */
.features-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1.2rem;
    margin-top: 2rem;
}
.feature-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 14px;
    padding: 1.6rem;
    transition: all 0.3s ease;
    display: flex;
    gap: 1rem;
    align-items: flex-start;
}
.feature-card:hover {
    background: rgba(99,102,241,0.08);
    border-color: rgba(99,102,241,0.4);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(99,102,241,0.12);
}
.feature-card .fc-icon {
    font-size: 1.6rem;
    min-width: 2.5rem;
    height: 2.5rem;
    background: rgba(99,102,241,0.12);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.feature-card .fc-text h4 {
    font-size: 0.92rem;
    font-weight: 600;
    color: #c7d2fe;
    margin-bottom: 0.3rem;
}
.feature-card .fc-text p {
    font-size: 0.82rem;
    color: #64748b;
    line-height: 1.6;
}

/* ── College Info Badge ── */
.college-badge {
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(16,185,129,0.08));
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 16px;
    padding: 1.6rem 2rem;
    text-align: center;
    margin: 2.5rem 0;
}
.college-badge h3 {
    font-family: 'Playfair Display', serif;
    color: #a5b4fc;
    font-size: 1.1rem;
    margin-bottom: 0.3rem;
}
.college-badge p { color: #64748b; font-size: 0.85rem; }

/* ── Viz Section ── */
.viz-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 16px;
    overflow: hidden;
    transition: all 0.3s ease;
}
.viz-card:hover {
    border-color: rgba(99,102,241,0.45);
    box-shadow: 0 10px 40px rgba(0,0,0,0.3);
}
.viz-card .viz-header {
    padding: 1rem 1.4rem;
    border-bottom: 1px solid rgba(99,102,241,0.15);
    background: rgba(99,102,241,0.06);
}
.viz-card .viz-header h4 {
    font-size: 0.82rem;
    color: #a5b4fc;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
}
.viz-card .viz-body { padding: 1rem; }

/* ── Prediction Form ── */
.form-section-title {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #6366f1;
    font-weight: 700;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(99,102,241,0.2);
}

/* Wrap the entire form in a card */
[data-testid="stForm"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 20px !important;
    padding: 2rem 2.2rem !important;
    box-shadow: 0 4px 40px rgba(0,0,0,0.3) !important;
}

/* Streamlit input overrides */
[data-testid="stSelectbox"] > div,
[data-testid="stNumberInput"] > div,
[data-testid="stTextInput"] > div {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(99,102,241,0.3) !important;
    border-radius: 10px !important;
}
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label,
[data-testid="stTextInput"] label {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
}

/* Submit Button */
.stFormSubmitButton > button, .stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.9rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.35) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stFormSubmitButton > button:hover, .stButton > button:hover {
    background: linear-gradient(135deg, #818cf8 0%, #6366f1 100%) !important;
    box-shadow: 0 6px 25px rgba(99,102,241,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── Result Box ── */
.result-wrapper {
    margin-top: 2rem;
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-wrapper.excellent {
    background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(5,150,105,0.06));
    border: 1px solid rgba(16,185,129,0.4);
}
.result-wrapper.good {
    background: linear-gradient(135deg, rgba(59,130,246,0.12), rgba(37,99,235,0.06));
    border: 1px solid rgba(59,130,246,0.4);
}
.result-wrapper.average {
    background: linear-gradient(135deg, rgba(245,158,11,0.12), rgba(217,119,6,0.06));
    border: 1px solid rgba(245,158,11,0.4);
}
.result-wrapper.poor {
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(220,38,38,0.06));
    border: 1px solid rgba(239,68,68,0.4);
}
.result-wrapper .result-score {
    font-family: 'Playfair Display', serif;
    font-size: 4rem;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.excellent .result-score  { color: #34d399; }
.good .result-score       { color: #60a5fa; }
.average .result-score    { color: #fbbf24; }
.poor .result-score       { color: #f87171; }

.result-wrapper .result-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: #64748b;
    margin-bottom: 1rem;
}
.result-wrapper .result-badge {
    display: inline-block;
    padding: 0.4rem 1.4rem;
    border-radius: 50px;
    font-size: 0.88rem;
    font-weight: 600;
    letter-spacing: 0.06em;
}
.excellent .result-badge  { background: rgba(16,185,129,0.2);  color: #34d399; border: 1px solid rgba(16,185,129,0.4); }
.good .result-badge       { background: rgba(59,130,246,0.2);  color: #60a5fa; border: 1px solid rgba(59,130,246,0.4); }
.average .result-badge    { background: rgba(245,158,11,0.2);  color: #fbbf24; border: 1px solid rgba(245,158,11,0.4); }
.poor .result-badge       { background: rgba(239,68,68,0.2);   color: #f87171; border: 1px solid rgba(239,68,68,0.4); }

.result-wrapper .result-hint {
    margin-top: 1.2rem;
    font-size: 0.85rem;
    color: #475569;
    line-height: 1.6;
}

/* ── Section Divider ── */
.section-divider {
    border: none;
    border-top: 1px solid rgba(99,102,241,0.15);
    margin: 2rem 0;
}

/* ── Streamlit overrides ── */
[data-testid="stVerticalBlock"] > * { color: #c4c9e2; }
.stMarkdown p { color: #94a3b8; }
footer { display: none !important; }
#MainMenu { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span class="brand-icon">🎓</span>
        <h2>Student Performance<br>Prediction System</h2>
        <p>Bhadrak Autonomous College</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠  Home","🎯  Predict Performance"],
        label_visibility="hidden"
    )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="padding: 1rem; text-align: center;">
        <p style="font-size:0.72rem; color:#374151; line-height:1.6;">
            Powered by<br>
            <span style="color:#6366f1; font-weight:600;">Linear Regression ML</span><br>
            <span style="color:#374151;">scikit-learn · streamlit</span>
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── Model Loader ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("models/linear_regression_model.pkl")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if "Home" in page:
    st.markdown("""
    <div class="page-header">
        <div class="eyebrow">Machine Learning Project · 2024–25</div>
        <h1>Student Performance<br><span>Prediction System</span></h1>
        <p>An intelligent ML-powered application to forecast academic outcomes using behavioral and academic indicators.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric Row ──
    st.markdown("""
    <div class="metric-row">
        <div class="metric-card" style="--accent: #6366f1;">
            <div class="metric-icon">🧠</div>
            <div class="metric-value">LR</div>
            <div class="metric-label">Model Type</div>
        </div>
        <div class="metric-card" style="--accent: #10b981;">
            <div class="metric-icon">📐</div>
            <div class="metric-value">11</div>
            <div class="metric-label">Input Features</div>
        </div>
        <div class="metric-card" style="--accent: #f59e0b;">
            <div class="metric-icon">🎯</div>
            <div class="metric-value">%</div>
            <div class="metric-label">Prediction Target</div>
        </div>
        <div class="metric-card" style="--accent: #ec4899;">
            <div class="metric-icon">⚡</div>
            <div class="metric-value">Live</div>
            <div class="metric-label">Inference</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── College Badge ──
    st.markdown("""
    <div class="college-badge">
        <h3>🏛️ Bhadrak Autonomous College, Odisha</h3>
        <p>Department of Computer Science &amp; Applications · Academic Year 2024–25</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Feature Cards ──
    st.markdown("""
    <div class="features-grid">
        <div class="feature-card">
            <div class="fc-icon">🔬</div>
            <div class="fc-text">
                <h4>Robust Preprocessing Pipeline</h4>
                <p>Automated feature encoding, standard scaling, and missing-value handling using scikit-learn pipelines.</p>
            </div>
        </div>
        <div class="feature-card">
            <div class="fc-icon">📊</div>
            <div class="fc-text">
                <h4>Visual Exploratory Analysis</h4>
                <p>Correlation matrices, distribution plots, and actual-vs-predicted scatter charts for full model transparency.</p>
            </div>
        </div>
        <div class="feature-card">
            <div class="fc-icon">⚡</div>
            <div class="fc-text">
                <h4>Real-Time Inference</h4>
                <p>Enter student details and receive instant percentage predictions with categorical performance grading.</p>
            </div>
        </div>
        <div class="feature-card">
            <div class="fc-icon">📁</div>
            <div class="fc-text">
                <h4>Serialized Model Artifact</h4>
                <p>Trained pipeline saved via joblib, enabling fast load-and-predict without retraining on every run.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)




# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif "Predict" in page:
    st.markdown("""
    <div class="page-header">
        <div class="eyebrow">Live Inference Engine</div>
        <h1>Predict Student <span>Performance</span></h1>
        <p>Fill in the student profile below to generate an instant performance prediction.</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        model = load_model()
    except FileNotFoundError:
        st.error("⚠️  Model file not found at `models/linear_regression_model.pkl`. Please run `train.py` first.")
        st.stop()

    with st.form("prediction_form", clear_on_submit=False):

        # ── Section 1: Academic Profile ──
        st.markdown('<div class="form-section-title">📚 Academic Profile</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3, gap="medium")

        with col1:
            course_type    = st.selectbox("Course Type", ["UG", "PG"])
            department     = st.selectbox("Department",
                                          ["Computer Science", "Physics", "Mathematics",
                                           "Botany", "Zoology", "Commerce", "Arts"])
            year_of_course = st.selectbox("Year of Course", [1, 2, 3])

        with col2:
            prev_sem1_sgpa = st.number_input("Previous Sem 1 SGPA", 0.0, 10.0, 7.5, step=0.1)
            prev_sem2_sgpa = st.number_input("Previous Sem 2 SGPA", 0.0, 10.0, 7.8, step=0.1)
            family_background = st.selectbox("Family Background", ["Urban", "Rural"])

        with col3:
            attendance      = st.number_input("Attendance (%)", 0.0, 100.0, 75.0, step=0.5)
            internal_perf   = st.number_input("Internal Performance (%)", 0.0, 100.0, 80.0, step=0.5)
            lab_perf        = st.number_input("Lab Performance (%)", 0.0, 100.0, 85.0, step=0.5)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Section 2: Behavioral Indicators ──
        st.markdown('<div class="form-section-title">🧩 Behavioral Indicators</div>', unsafe_allow_html=True)
        col4, col5, col6 = st.columns([1, 1, 1], gap="medium")

        with col4:
            teacher_review = st.number_input("Teacher Review (out of 10)", 1.0, 10.0, 8.0, step=0.1)
        with col5:
            study_hours    = st.number_input("Study Hours per Day", 0.0, 24.0, 4.0, step=0.5)
        with col6:
            st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("⚡  Generate Prediction")

    # ── Result ──
    if submitted:
        input_data = {
            "Course_Type":              course_type,
            "Year_of_Course":           year_of_course,
            "Department":               department,
            "Attendance_%":             attendance,
            "Internal_Performance_%":   internal_perf,
            "Lab_Performance_%":        lab_perf,
            "Previous_Sem1_SGPA":       prev_sem1_sgpa,
            "Previous_Sem2_SGPA":       prev_sem2_sgpa,
            "Teacher_Review_out_of_10": teacher_review,
            "Study_Hours_per_day":      study_hours,
            "Family_Background":        family_background,
        }

        input_df = pd.DataFrame([input_data])

        try:
            prediction = model.predict(input_df)[0]
            prediction = max(0.0, min(100.0, prediction))  # clamp to [0, 100]

            if prediction >= 80:
                category  = "Excellent"
                css_class = "excellent"
                bar_color = "#34d399"
                hint      = "Outstanding academic performance — the student is on track for distinction."
            elif prediction >= 60:
                category  = "Good"
                css_class = "good"
                bar_color = "#60a5fa"
                hint      = "Solid performance. With consistent effort, further improvement is achievable."
            elif prediction >= 40:
                category  = "Average"
                css_class = "average"
                bar_color = "#fbbf24"
                hint      = "Performance is satisfactory. Focused revision and better study habits are recommended."
            else:
                category  = "Needs Improvement"
                css_class = "poor"
                bar_color = "#f87171"
                hint      = "Performance is below expectations. Intervention and additional support are advised."

            st.markdown(f"""
            <div class="result-wrapper {css_class}">
                <div class="result-label">Predicted Secured Percentage</div>
                <div class="result-score">{prediction:.1f}%</div>
                <div style="margin: 1rem auto; max-width: 300px; height: 6px; background: rgba(255,255,255,0.08); border-radius: 99px; overflow: hidden;">
                    <div style="width:{prediction:.1f}%; height:100%; border-radius:99px; background: {bar_color};"></div>
                </div>
                <div class="result-badge">{category}</div>
                <div class="result-hint">{hint}</div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")



