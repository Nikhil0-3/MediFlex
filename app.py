import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page configuration
st.set_page_config(
    page_title="MediFlex - Medicine Recommendation System",
    page_icon="💊",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #B8EAFF;
        padding: 2rem;
    }
    .stApp {
        background: linear-gradient(to bottom, #f8f9fa 0%, #e9ecef 100%);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 18px;
        font-weight: 600;
        padding: 12px 28px;
        border-radius: 12px;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    .medicine-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border: 2px solid #e9ecef;
        transition: all 0.3s ease;
    }
    .medicine-card:hover {
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
        border-color: #667eea;
    }
    .header-text {
        color: #667eea;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    .subheader-text {
        color: #6c757d;
        font-size: 1.4rem;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .confidence-high {
        color: #10b981;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .confidence-medium {
        color: #f59e0b;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .confidence-low {
        color: #ef4444;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .top-recommendation {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        font-size: 1.3rem;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 8px 16px rgba(16, 185, 129, 0.3);
        margin: 20px 0;
    }
    .symptom-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        display: inline-block;
        margin: 5px;
        font-weight: 500;
    }
    .info-box {
        background: #f8f9ff;
        padding: 1.25rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .section-box {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin: 1.5rem 0;
    }
    .stTextArea textarea {
        border: 2px solid #e9ecef;
        border-radius: 10px;
        font-size: 1rem;
        background-color: #f8f9ff !important;
    }
    .stTextArea textarea::placeholder {
        color: #6c757d !important;
        opacity: 1 !important;
    }
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.15);
        background-color: #ffffff !important;
    }
    .stRadio > label {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    .stRadio > div {
        color: #000000 !important;
    }
    .stRadio div[role="radiogroup"] label {
        color: #000000 !important;
        font-weight: 500 !important;
    }
    .stTextArea > label {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    .stTextArea textarea {
        color: #2009C8 !important;
    }
    .stMultiSelect > label {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    .stMultiSelect div[data-baseweb="select"] {
        background-color: #f8f9ff !important;
    }
    .stMultiSelect div[data-baseweb="select"] > div {
        background-color: #f8f9ff !important;
        border: 2px solid #e9ecef !important;
        border-radius: 10px !important;
    }
    .stMultiSelect div[data-baseweb="select"] span,
    .stMultiSelect div[data-baseweb="select"] div,
    .stMultiSelect div[data-baseweb="select"] p {
        color: #000000 !important;
        font-weight: 500 !important;
    }
    .stMultiSelect [role="button"] {
        color: #000000 !important;
    }
    .stMultiSelect [data-baseweb="popover"] span {
        color: #000000 !important;
    }
    .stMultiSelect div[data-baseweb="select"]:focus-within > div {
        border-color: #667eea !important;
        background-color: #ffffff !important;
    }
    .stMultiSelect input {
        color: #2009C8 !important;
    }
    .stMultiSelect div[data-baseweb="tag"] {
        background-color: #667eea !important;
        color: white !important;
    }
    .stSidebar {
        background-color: #f8f9fa;
    }
    .stSidebar .stMarkdown {
        color: #2009C8 !important;
    }
    .stSidebar h3 {
        color: #667eea !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and preprocessors
@st.cache_resource
def load_model_and_data():
    model = tf.keras.models.load_model("medicine_model.h5", compile=False)
    with open("tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    with open("medicine_labels.pkl", "rb") as handle:
        medicine_list = pickle.load(handle)
    return model, tokenizer, medicine_list

# Header
st.markdown('<p class="header-text">💊 MediFlex</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader-text">Intelligent Medicine Recommendation System</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; margin-bottom: 1.5rem;'>
            <h1 style='color: white; margin: 0; font-size: 1.8rem;'>💊 MediFlex</h1>
            <p style='color: #e0e7ff; margin: 0.5rem 0 0 0; font-size: 1rem;'>Smart Health Assistant</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📖 About")
    st.markdown("""
    <div class='info-box'>
        <p style='margin: 0; color: #495057; line-height: 1.8; font-size: 0.95rem;'>
        MediFlex uses advanced machine learning to recommend appropriate medicines based on your symptoms.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📝 How to Use")
    st.markdown("""
    1. 🔍 Enter or select symptoms
    2. 🎯 Adjust confidence level
    3. 🔬 Get recommendations
    4. 💊 Consult healthcare provider
    """)
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold (%)",
        min_value=10,
        max_value=100,
        value=40,
        step=5,
        help="Minimum confidence level for medicine recommendations"
    )
    
    max_results = st.number_input(
        "Maximum Results",
        min_value=3,
        max_value=20,
        value=10,
        step=1,
        help="Number of top recommendations to display"
    )
    
    show_all = st.checkbox("Show all predictions", value=False)
    
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #667eea; font-size: 0.85rem;'>
            <p>⚕️ Always consult a healthcare professional</p>
        </div>
    """, unsafe_allow_html=True)

# Main content

# Input methods inside the section box
col_radio1, col_radio2, col_radio3 = st.columns([1, 2, 1])
with col_radio2:
    st.markdown("<style>div[role='radiogroup'] label p {color: #667eea !important; font-weight: 600 !important;}</style>", unsafe_allow_html=True)
    input_method = st.radio(
        "Choose input method:",
        ["✍️ Text Input", "📋 Select from List"],
        horizontal=True
    )

if input_method == "✍️ Text Input":
    user_symptoms = st.text_area(
        "🩺 Enter Your Symptoms",
        placeholder="Example: fever, headache, cough, sore throat, body pain",
        height=120,
        help="Enter symptoms separated by commas or spaces"
    )
    
    # Show entered symptoms as badges
    if user_symptoms.strip():
        st.markdown("<div style='margin-top: 1rem;'><strong style='color: #495057;'>Your Symptoms:</strong></div>", unsafe_allow_html=True)
        symptoms_list = [s.strip() for s in user_symptoms.replace(',', ' ').split() if s.strip()]
        badge_html = "<div style='margin-top: 0.5rem;'>" + "".join([f'<span class="symptom-badge">{sym}</span>' for sym in symptoms_list[:10]]) + "</div>"
        st.markdown(badge_html, unsafe_allow_html=True)
else:
    common_symptoms = [
        "Fever", "Headache", "Cough", "Cold", "Sore Throat",
        "Body Pain", "Nausea", "Vomiting", "Diarrhea", "Fatigue",
        "Runny Nose", "Sneezing", "Chest Pain", "Shortness of Breath",
        "Stomach Pain", "Dizziness", "Weakness", "Muscle Ache"
    ]
    selected_symptoms = st.multiselect(
        "Select your symptoms:",
        common_symptoms,
        help="Choose one or more symptoms from the list"
    )
    user_symptoms = ", ".join(selected_symptoms)
    
    # Show selected count
    if selected_symptoms:
        st.success(f"✅ Selected {len(selected_symptoms)} symptom(s)")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_button = st.button("🔍 Get Recommendations", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Tips section below input
col_tip1, col_tip2 = st.columns(2)

with col_tip1:
    st.markdown("""
        <div style='background: #f8f9ff; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #667eea; height: 100%;'>
            <h4 style='color: #667eea; margin-top: 0; font-size: 1.1rem;'>💡 Quick Tips</h4>
            <ul style='color: #495057; line-height: 1.8; margin: 0; padding-left: 1.5rem; font-size: 0.95rem;'>
                <li>Be specific about symptoms</li>
                <li>Mention all relevant issues</li>
                <li>Include severity if known</li>
                <li>Note symptom duration</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

with col_tip2:
    st.markdown("""
        <div style='background: #fff4e6; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #f59e0b; height: 100%;'>
            <h4 style='color: #f59e0b; margin-top: 0; font-size: 1.1rem;'>⚠️ Important Notice</h4>
            <p style='color: #6c757d; margin: 0; line-height: 1.8; font-size: 0.95rem;'>
                This system is for educational purposes only. 
                Always seek professional medical advice for diagnosis and treatment.
            </p>
        </div>
    """, unsafe_allow_html=True)

# Prediction
if predict_button:
    if user_symptoms.strip():
        with st.spinner("Analyzing symptoms..."):
            try:
                model, tokenizer, medicine_list = load_model_and_data()
                
                # Preprocess input
                sequence = tokenizer.texts_to_sequences([user_symptoms])
                padded_sequence = pad_sequences(sequence, maxlen=5)
                
                # Predict
                predictions = model.predict(padded_sequence, verbose=0)[0]
                
                # Create results dataframe
                results = []
                for i, medicine in enumerate(medicine_list):
                    confidence = predictions[i] * 100
                    if show_all or confidence >= confidence_threshold:
                        results.append({
                            'medicine': medicine,
                            'confidence': confidence
                        })
                
                # Sort by confidence
                results = sorted(results, key=lambda x: x['confidence'], reverse=True)
                
                # Limit results if not showing all
                if not show_all:
                    results = results[:max_results]
                
                st.markdown("---")
                
                if results:
                    # Display top recommendation prominently
                    top_med = results[0]
                    st.markdown(f"""
                        <div class='top-recommendation'>
                            <div style='font-size: 1.1rem; margin-bottom: 0.5rem;'>🏆 Top Recommendation</div>
                            <div style='font-size: 2rem; font-weight: 800; margin-bottom: 0.3rem;'>{top_med['medicine']}</div>
                            <div style='font-size: 1.3rem; opacity: 0.95;'>{top_med['confidence']:.1f}% Match Confidence</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Statistics
                    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("📊 Total Results", len(results))
                    with col_stat2:
                        high_conf = len([r for r in results if r['confidence'] >= 70])
                        st.metric("✅ High Confidence", high_conf)
                    with col_stat3:
                        avg_conf = sum(r['confidence'] for r in results) / len(results)
                        st.metric("📈 Average Match", f"{avg_conf:.1f}%")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Display all recommendations
                    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
                    st.markdown("""
                        <div class='section-box'>
                            <h3 style='color: #667eea; margin: 0; font-size: 1.6rem;'>📋 Detailed Recommendations</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    for idx, result in enumerate(results, 1):
                        confidence = result['confidence']
                        
                        # Determine confidence level styling
                        if confidence >= 70:
                            conf_class = "confidence-high"
                            emoji = "🟢"
                            bar_color = "#10b981"
                        elif confidence >= 40:
                            conf_class = "confidence-medium"
                            emoji = "🟡"
                            bar_color = "#f59e0b"
                        else:
                            conf_class = "confidence-low"
                            emoji = "🟠"
                            bar_color = "#ef4444"
                        
                        # Create medicine card
                        st.markdown(f"""
                            <div class='medicine-card'>
                                <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;'>
                                    <div style='flex: 1; min-width: 200px;'>
                                        <span style='background: #667eea; color: white; padding: 0.25rem 0.75rem; border-radius: 6px; font-weight: 600; font-size: 0.9rem;'>#{idx}</span>
                                        <span style='color: #212529; font-weight: 600; font-size: 1.15rem; margin-left: 0.75rem;'>{result['medicine']}</span>
                                    </div>
                                    <div style='margin-top: 0.5rem;'>
                                        <span style='font-size: 1.5rem;'>{emoji}</span> <span class='{conf_class}' style='font-size: 1.1rem;'>{confidence:.1f}%</span>
                                    </div>
                                </div>
                                <div style='margin-top: 1rem;'>
                                    <div style='background: #e9ecef; border-radius: 10px; height: 8px; overflow: hidden;'>
                                        <div style='background: {bar_color}; height: 100%; width: {confidence}%; transition: width 0.3s ease;'></div>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Disclaimer
                    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
                    st.markdown("""
                        <div style='background: #fff5f5; padding: 2rem; border-radius: 12px; border: 2px solid #fc8181;'>
                            <h3 style='color: #c53030; margin-top: 0; font-size: 1.4rem;'>⚠️ Medical Disclaimer</h3>
                            <p style='color: #742a2a; margin: 0; line-height: 1.9; font-size: 1rem;'>
                                These recommendations are generated by a machine learning system and should <strong>not replace professional medical advice</strong>.
                                Please consult with a qualified healthcare provider before taking any medication.
                                This tool is for educational and informational purposes only.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.markdown(f"""
                        <div style='background: #fef3c7; padding: 25px; border-radius: 15px; border-left: 5px solid #f59e0b;'>
                            <h3 style='color: #d97706; margin-top: 0;'>⚠️ No Results Found</h3>
                            <p style='color: #b45309; margin: 0;'>
                                No medicines found with confidence above {confidence_threshold}%. 
                                Try lowering the threshold in settings or enabling "Show all predictions".
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f"""
                    <div style='background: #fee2e2; padding: 25px; border-radius: 15px; border-left: 5px solid #ef4444;'>
                        <h3 style='color: #dc2626; margin-top: 0;'>❌ Error Occurred</h3>
                        <p style='color: #991b1b; margin: 0;'>
                            {str(e)}
                        </p>
                        <p style='color: #991b1b; margin-top: 10px; font-size: 0.9rem;'>
                            Please make sure the model and required files are in the correct location.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style='background: #fef3c7; padding: 25px; border-radius: 15px; border-left: 5px solid #f59e0b; text-align: center;'>
                <h3 style='color: #d97706; margin-top: 0;'>⚠️ No Symptoms Entered</h3>
                <p style='color: #b45309; margin: 0;'>
                    Please enter or select your symptoms before getting recommendations.
                </p>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<div style='margin: 3rem 0;'></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2.5rem; border-radius: 12px; text-align: center; color: white;'>
        <h2 style='margin: 0; font-size: 2rem; font-weight: 700;'>💊 MediFlex</h2>
        <p style='margin: 0.75rem 0; font-size: 1.1rem; color: #e0e7ff; font-weight: 500;'>Powered by Machine Learning & TensorFlow</p>
        <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #c7d2fe;'>Version 1.0 | For educational purposes only</p>
    </div>
    """,
    unsafe_allow_html=True
)