import streamlit as st
import pandas as pd
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Heart Health Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(255, 80, 80, 0.08), transparent 25%),
        radial-gradient(circle at 90% 20%, rgba(255, 0, 80, 0.06), transparent 25%),
        linear-gradient(135deg, #fff7f8 0%, #ffffff 45%, #fff5f6 100%);
}

/* Remove top padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Header */
.hero {
    background: linear-gradient(135deg, #8B0000, #d90429, #ef233c);
    padding: 40px 45px;
    border-radius: 25px;
    color: white;
    margin-bottom: 30px;
    box-shadow: 0 15px 35px rgba(217, 4, 41, 0.25);
    position: relative;
    overflow: hidden;
}

.hero:after {
    content: "♥";
    position: absolute;
    right: 45px;
    top: 5px;
    font-size: 150px;
    opacity: 0.08;
}

.hero h1 {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 10px;
}

.hero p {
    font-size: 17px;
    opacity: 0.92;
    max-width: 700px;
}

/* Section headings */
.section-title {
    font-size: 24px;
    font-weight: 800;
    color: #7f1d1d;
    margin-top: 15px;
    margin-bottom: 18px;
}

/* Cards */
.card {
    background: rgba(255, 255, 255, 0.92);
    padding: 24px;
    border-radius: 20px;
    border: 1px solid rgba(220, 38, 38, 0.08);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.06);
    margin-bottom: 20px;
}

.card-title {
    font-size: 17px;
    font-weight: 700;
    color: #7f1d1d;
    margin-bottom: 5px;
}

/* Metric cards */
.metric-card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid #fee2e2;
    box-shadow: 0 6px 18px rgba(0,0,0,0.05);
}

.metric-icon {
    font-size: 30px;
}

.metric-value {
    font-size: 25px;
    font-weight: 800;
    color: #991b1b;
}

.metric-label {
    font-size: 13px;
    color: #6b7280;
}

/* Predict button */
.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 15px;
    border: none;
    background: linear-gradient(135deg, #b91c1c, #ef233c);
    color: white;
    font-size: 18px;
    font-weight: 700;
    box-shadow: 0 8px 20px rgba(185, 28, 28, 0.25);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 25px rgba(185, 28, 28, 0.35);
}

/* Inputs */
div[data-baseweb="select"] > div {
    border-radius: 12px;
}

.stSlider > div {
    padding-top: 5px;
}

/* Result cards */
.result-high {
    background: linear-gradient(135deg, #fff1f2, #ffe4e6);
    border: 2px solid #ef4444;
    padding: 30px;
    border-radius: 22px;
    text-align: center;
    margin-top: 25px;
}

.result-low {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 2px solid #22c55e;
    padding: 30px;
    border-radius: 22px;
    text-align: center;
    margin-top: 25px;
}

.result-icon {
    font-size: 55px;
}

.result-title {
    font-size: 30px;
    font-weight: 800;
}

.result-text {
    font-size: 16px;
    color: #4b5563;
}

/* Info boxes */
.info-box {
    background: #fff7ed;
    border-left: 5px solid #f97316;
    padding: 18px;
    border-radius: 12px;
    margin-top: 20px;
}

.disclaimer {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    padding: 20px;
    border-radius: 15px;
    margin-top: 30px;
    font-size: 13px;
    color: #64748b;
}

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 13px;
    margin-top: 35px;
}

/* Hide Streamlit branding */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HERO HEADER
# =========================================================

st.markdown("""
<div class="hero">

<h1>🫀 Heart Health Predictor</h1>

<p>
AI-powered heart disease risk assessment using a trained
Machine Learning model. Enter your health parameters below
to receive an instant prediction.
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# INTRO CARDS
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-icon">🧠</div>
        <div class="metric-value">ML</div>
        <div class="metric-label">Machine Learning Powered</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-icon">⚡</div>
        <div class="metric-value">Instant</div>
        <div class="metric-label">Prediction Result</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-icon">📊</div>
        <div class="metric-value">10+</div>
        <div class="metric-label">Health Parameters</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# INPUT SECTION
# =========================================================

st.markdown(
    '<div class="section-title">👤 Personal Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider(
        "Age",
        18,
        100,
        40
    )

with col2:
    sex = st.selectbox(
        "Sex",
        ["M", "F"]
    )

with col3:
    resting_bp = st.number_input(
        "Resting Blood Pressure",
        min_value=80,
        max_value=200,
        value=120,
        step=1
    )


# =========================================================
# HEART PARAMETERS
# =========================================================

st.markdown(
    '<div class="section-title">❤️ Heart & Blood Parameters</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    cholesterol = st.number_input(
        "Cholesterol (mg/dL)",
        min_value=100,
        max_value=600,
        value=200,
        step=1
    )

with col2:
    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col3:
    max_hr = st.slider(
        "Maximum Heart Rate",
        60,
        220,
        150
    )


# =========================================================
# ECG PARAMETERS
# =========================================================

st.markdown(
    '<div class="section-title">🩺 ECG & Exercise Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"],
        format_func=lambda x: {
            "ATA": "Atypical Angina",
            "NAP": "Non-Anginal Pain",
            "TA": "Typical Angina",
            "ASY": "Asymptomatic"
        }[x]
    )

with col2:
    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

with col3:
    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        ["Y", "N"],
        format_func=lambda x: "Yes" if x == "Y" else "No"
    )


# =========================================================
# ST PARAMETERS
# =========================================================

st.markdown(
    '<div class="section-title">📈 ST Segment Parameters</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    oldpeak = st.slider(
        "Oldpeak (ST Depression)",
        0.0,
        6.0,
        1.0,
        step=0.1
    )

with col2:
    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"],
        format_func=lambda x: {
            "Up": "Upsloping",
            "Flat": "Flat",
            "Down": "Downsloping"
        }[x]
    )


# =========================================================
# INPUT SUMMARY
# =========================================================

with st.expander("🔍 Review Your Information"):

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.write(f"**Age:** {age}")
        st.write(f"**Sex:** {sex}")
        st.write(f"**Resting BP:** {resting_bp} mm Hg")
        st.write(f"**Cholesterol:** {cholesterol} mg/dL")
        st.write(f"**Fasting Blood Sugar:** {'Yes' if fasting_bs else 'No'}")

    with summary_col2:
        st.write(f"**Maximum Heart Rate:** {max_hr}")
        st.write(f"**Chest Pain:** {chest_pain}")
        st.write(f"**Resting ECG:** {resting_ecg}")
        st.write(f"**Exercise Angina:** {'Yes' if exercise_angina == 'Y' else 'No'}")
        st.write(f"**Oldpeak:** {oldpeak}")
        st.write(f"**ST Slope:** {st_slope}")


# =========================================================
# PREDICTION
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

predict_col, reset_col = st.columns([3, 1])

with predict_col:

    predict_button = st.button(
        "🫀 ANALYZE HEART HEALTH",
        use_container_width=True
    )

with reset_col:

    if st.button(
        "🔄 Reset",
        use_container_width=True
    ):
        st.rerun()


if predict_button:

    # -----------------------------------------------------
    # CREATE RAW INPUT
    # -----------------------------------------------------

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,

        'Sex_' + sex: 1,

        'ChestPainType_' + chest_pain: 1,

        'RestingECG_' + resting_ecg: 1,

        'ExerciseAngina_' + exercise_angina: 1,

        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])


    # -----------------------------------------------------
    # MATCH MODEL COLUMNS
    # -----------------------------------------------------

    for col in expected_columns:

        if col not in input_df.columns:
            input_df[col] = 0


    input_df = input_df[expected_columns]


    # -----------------------------------------------------
    # SCALE
    # -----------------------------------------------------

    scaled_input = scaler.transform(input_df)


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    prediction = model.predict(scaled_input)[0]


    # -----------------------------------------------------
    # PROBABILITY
    # -----------------------------------------------------

    probability = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(scaled_input)

        probability = probabilities[0][1] * 100


    # =====================================================
    # RESULT
    # =====================================================

    st.markdown(
        '<div class="section-title">📋 Prediction Result</div>',
        unsafe_allow_html=True
    )


    if prediction == 1:

        st.markdown("""
        
                The model has identified patterns in the provided
                health parameters that are associated with a higher
                risk of heart disease.
         
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        
                Based on the provided information, the model predicts


                A LOWER RISK OF HEART DISEASE.
           

        </div>
        """, unsafe_allow_html=True)


    # =====================================================
    # PROBABILITY DISPLAY
    # =====================================================

    if probability is not None:

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            f"""
            
                    📊 Model Risk Probability
                
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(
            min(int(probability), 100)
        )


    # =====================================================
    # RECOMMENDATION
    # =====================================================

    if prediction == 1:

        st.markdown("""
        <div class="info-box">

        <b>⚠️ What should you do?</b>

        <br><br>

        This prediction is generated by a machine learning model
        and should not be considered a medical diagnosis.

        If you have symptoms or concerns about your heart health,
        consult a qualified healthcare professional.

        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="info-box">

        <b>💡 Keep taking care of your heart</b>

        <br><br>

        Continue maintaining a healthy lifestyle through regular
        physical activity, balanced nutrition, adequate sleep,
        and routine health checkups.

        </div>
        """, unsafe_allow_html=True)


# =========================================================
# DISCLAIMER
# =========================================================

st.markdown("""
<div class="disclaimer">

<b>⚕️ Medical Disclaimer</b>

<br><br>

This application is an educational machine learning project.
The prediction is based on a trained statistical model and
should NOT be used as a substitute for professional medical
advice, diagnosis, or treatment.

If you are experiencing chest pain, difficulty breathing,
fainting, or other serious symptoms, seek immediate medical
attention.

</div>
""", unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

🫀 Heart Health Predictor &nbsp; • &nbsp;
Machine Learning Project &nbsp; • &nbsp;
Built with Python + Streamlit

<br><br>

Made by <b>Akarsh</b>

</div>
""", unsafe_allow_html=True)