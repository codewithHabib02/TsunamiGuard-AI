import streamlit as st 
import pandas as pd 
import joblib 
 
 
# ========================================================= 
# PAGE CONFIG 
# ========================================================= 
 
st.set_page_config( 
    page_title="TsunamiGuard AI", 
    page_icon="🌊", 
    layout="wide", 
    initial_sidebar_state="expanded" 
) 
 
 
# ========================================================= 
# LOAD MODEL 
# ========================================================= 
 
@st.cache_resource 
def load_model(): 
    return joblib.load("tsunami_model.pkl") 
 
 
model = load_model() 
 
 
# ========================================================= 
# CUSTOM CSS 
# ========================================================= 
 
st.markdown(""" 
<style> 
 
    /* Main background */ 
    .stApp { 
        background: 
            linear-gradient( 
                135deg, 
                #06141f 0%, 
                #082b3a 45%, 
                #063f50 100% 
            ); 
        color: white; 
    } 
 
    /* Main content */ 
    .block-container { 
        padding-top: 2rem; 
        padding-bottom: 3rem; 
        max-width: 1200px; 
    } 
 
    /* Header */ 
    .main-title { 
        text-align: center; 
        font-size: 48px; 
        font-weight: 800; 
        margin-bottom: 5px; 
        color: white; 
    } 
 
    .subtitle { 
        text-align: center; 
        font-size: 18px; 
        color: #b8dce5; 
        margin-bottom: 35px; 
    } 
 
    /* Cards */ 
    .card { 
        background: rgba(255, 255, 255, 0.08); 
        border: 1px solid rgba(255, 255, 255, 0.15); 
        border-radius: 18px; 
        padding: 25px; 
        margin-bottom: 20px; 
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25); 
    } 
 
    .card-title { 
        font-size: 23px; 
        font-weight: 700; 
        color: white; 
        margin-bottom: 15px; 
    } 
 
    /* Input labels */ 
    label { 
        color: #e8f7fa !important; 
        font-weight: 600 !important; 
    } 
 
    /* Input boxes */ 
    div[data-baseweb="input"] { 
        background-color: rgba(255, 255, 255, 0.95); 
        border-radius: 10px; 
    } 
 
    input { 
        color: #102027 !important; 
    } 
 
    /* Button */ 
    .stButton > button { 
        width: 100%; 
        border-radius: 12px; 
        height: 50px; 
        font-size: 18px; 
        font-weight: 700; 
        background: #0ea5a8; 
        color: white; 
        border: none; 
        transition: 0.3s; 
    } 
 
    .stButton > button:hover { 
        background: #14b8a6; 
        transform: translateY(-2px); 
    } 
 
    /* Prediction cards */ 
    .prediction-danger { 
        background: rgba(220, 38, 38, 0.18); 
        border: 1px solid rgba(248, 113, 113, 0.5); 
        border-radius: 18px; 
        padding: 25px; 
        text-align: center; 
    } 
 
    .prediction-safe { 
        background: rgba(16, 185, 129, 0.18); 
        border: 1px solid rgba(52, 211, 153, 0.5); 
        border-radius: 18px; 
        padding: 25px; 
        text-align: center; 
    } 
 
    .prediction-title { 
        font-size: 30px; 
        font-weight: 800; 
        margin-bottom: 8px; 
    } 
 
    .probability { 
        font-size: 22px; 
        font-weight: 600; 
    } 
 
    /* Footer */ 
    .footer { 
        text-align: center; 
        color: #9ccbd4; 
        font-size: 14px; 
        margin-top: 40px; 
    } 
 
    /* Sidebar */ 
    section[data-testid="stSidebar"] { 
        background: #041923; 
    } 
 
</style> 
""", unsafe_allow_html=True) 
 
 
# ========================================================= 
# HEADER 
# ========================================================= 
 
st.markdown( 
    '<div class="main-title">🌊 TsunamiGuard AI</div>', 
    unsafe_allow_html=True 
) 
 
st.markdown( 
    '<div class="subtitle">' 
    'Machine Learning–Powered Tsunami Risk Prediction' 
    '</div>', 
    unsafe_allow_html=True 
) 
 
 
# ========================================================= 
# SIDEBAR 
# ========================================================= 
 
with st.sidebar: 
 
    st.markdown("## 🌊 TsunamiGuard AI") 
 
    st.markdown(""" 
    This application uses a machine learning model to predict 
    whether an earthquake event is associated with tsunami activity. 
 
    **Model Inputs** 
 
    • Year   
    • Distance to nearest station   
    • Longitude   
    • Number of stations   
    • Latitude   
    • Depth   
    • Azimuthal gap   
    • Significance 
    """) 
 
    st.markdown("---") 
 
    st.info( 
       "This application is an educational machine learning project for tsunami prediction."

    ) 
 
 
# ========================================================= 
# INPUT SECTION 
# ========================================================= 
 
st.markdown( 
    '<div class="card">', 
    unsafe_allow_html=True 
) 
 
st.markdown( 
    '<div class="card-title">📊 Earthquake Information</div>', 
    unsafe_allow_html=True 
) 
 
col1, col2 = st.columns(2) 
 
 
with col1: 
 
    year = st.number_input( 
        "Year", 
        min_value=1900, 
        max_value=2100, 
        value=2024, 
        step=1 
    ) 
 
    dmin = st.number_input( 
        "Minimum Distance (dmin)", 
        min_value=0.0, 
        value=0.0, 
        step=0.1, 
        format="%.4f" 
    ) 
 
    longitude = st.number_input( 
        "Longitude", 
        min_value=-180.0, 
        max_value=180.0, 
        value=0.0, 
        step=0.1 
    ) 
 
    nst = st.number_input( 
        "Number of Stations (nst)", 
        min_value=0, 
        value=10, 
        step=1 
    ) 
 
 
with col2: 
 
    latitude = st.number_input( 
        "Latitude", 
        min_value=-90.0, 
        max_value=90.0, 
        value=0.0, 
        step=0.1 
    ) 
 
    depth = st.number_input( 
        "Depth (km)", 
        min_value=0.0, 
        value=10.0, 
        step=0.1 
    ) 
 
    gap = st.number_input( 
        "Azimuthal Gap", 
        min_value=0.0, 
        max_value=360.0, 
        value=180.0, 
        step=1.0 
    ) 
 
    sig = st.number_input( 
        "Significance (sig)", 
        min_value=0, 
        value=100, 
        step=1 
    ) 
 
st.markdown("</div>", unsafe_allow_html=True) 
 
 
# ========================================================= 
# PREDICTION BUTTON 
# ========================================================= 
 
st.write("") 
 
predict_button = st.button( 
    "🌊 Predict Tsunami Risk" 
) 
 
 
# ========================================================= 
# PREDICTION 
# ========================================================= 
 
if predict_button: 
 
    input_data = pd.DataFrame({ 
        "Year": [year], 
        "dmin": [dmin], 
        "longitude": [longitude], 
        "nst": [nst], 
        "latitude": [latitude], 
        "depth": [depth], 
        "gap": [gap], 
        "sig": [sig] 
    }) 
 
    try: 
 
        prediction = model.predict(input_data)[0] 
 
        # Get probability 
        probabilities = model.predict_proba(input_data)[0] 
 
        # Find probability for class 1 
        if hasattr(model, "classes_"): 
            classes = model.classes_ 
            tsunami_index = list(classes).index(1) 
            tsunami_probability = probabilities[tsunami_index] 
        else: 
            tsunami_probability = probabilities[1] 
 
        tsunami_probability_percent = tsunami_probability * 100 
 
 
        # ================================================= 
        # RESULT 
        # ================================================= 
 
        st.markdown("---") 
 
        st.markdown( 
            '<div class="card-title">🔍 Prediction Result</div>', 
            unsafe_allow_html=True 
        ) 
 
        if prediction == 1: 
 
            st.markdown( 
                f""" 
                <div class="prediction-danger"> 
                    <div class="prediction-title"> 
                        ⚠️ Yes — Tsunami Predicted 
                    </div> 
                    <div class="probability"> 
                        Probability of tsunami occurrence: {tsunami_probability_percent:.2f}% 
                    </div> 
                </div> 
                """, 
                unsafe_allow_html=True 
            ) 
 
        else: 
 
            st.markdown( 
                f""" 
                <div class="prediction-safe"> 
                    <div class="prediction-title"> 
                        ✅ No — Tsunami Not Predicted 
                    </div> 
                    <div class="probability"> 
                        Probability of tsunami occurrence: {tsunami_probability_percent:.2f}% 
                    </div> 
                </div> 
                """, 
                unsafe_allow_html=True 
            ) 
 
 
        # ================================================= 
        # INPUT SUMMARY 
        # ================================================= 
 
        st.write("") 
 
        st.markdown( 
            '<div class="card">', 
            unsafe_allow_html=True 
        ) 
 
        st.markdown( 
            '<div class="card-title">📋 Input Summary</div>', 
            unsafe_allow_html=True 
        ) 
 
        st.dataframe( 
            input_data, 
            use_container_width=True, 
            hide_index=True 
        ) 
 
        st.markdown("</div>", unsafe_allow_html=True) 
 
 
    except Exception as e: 
 
        st.error( 
            f"Prediction error: {e}" 
        ) 
 
 
# ========================================================= 
# FOOTER 
# ========================================================= 
 
st.markdown( 
    """ 
    <div class="footer"> 
       Developed By Habibulie 🟢 alias Leda🔴 | Data Science Student.
    </div> 
    """, 
    unsafe_allow_html=True 
)