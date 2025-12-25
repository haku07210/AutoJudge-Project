import streamlit as st
import joblib
import os

# --- 1. Smart Model Loader (Fixes Error 13) ---
@st.cache_resource
def load_models():
    # List of possible locations for your models (Professional -> Backup)
    # This ensures it works whether files are in 'models/' or root.
    base_dir = os.path.dirname(__file__)
    
    locations = [
        os.path.join(base_dir, 'models'),  # Check 'models' folder first
        base_dir                           # Check main folder as backup
    ]
    
    clf, reg, vectorizer = None, None, None
    
    # Helper to find and load a specific file
    def find_and_load(filename):
        for loc in locations:
            path = os.path.join(loc, filename)
            # Check if file exists AND is actually a file (not a folder)
            if os.path.exists(path) and os.path.isfile(path):
                try:
                    return joblib.load(path)
                except Exception:
                    continue # Try next location if load fails
        return None

    clf = find_and_load('model_class.pkl')
    reg = find_and_load('model_score.pkl')
    vectorizer = find_and_load('tfidf.pkl')
    
    return clf, reg, vectorizer

# Load them now
clf, reg, vectorizer = load_models()

# --- 2. Configure the Page ---
st.set_page_config(page_title="AutoJudge AI", page_icon="🤖", layout="centered")

st.title("🤖 AutoJudge: Difficulty Predictor")
st.markdown("### Professional AI System")
st.markdown("Enter the problem details below to predict its difficulty using Machine Learning.")

# --- 3. Input Fields ---
col1, col2 = st.columns(2)

with col1:
    st.info("1. Problem Description")
    desc_input = st.text_area("Paste the main task here:", height=150, placeholder="e.g., Write a program to find the shortest path...")

with col2:
    st.info("2. Input Description")
    inp_input = st.text_area("Paste input constraints:", height=70, placeholder="e.g., The first line contains integer N...")
    
    st.info("3. Output Description")
    out_input = st.text_area("Paste expected output:", height=70, placeholder="e.g., Print the sum of the array...")

# --- 4. Prediction Logic ---
if st.button("🚀 Predict Difficulty", use_container_width=True):
    if clf is None or vectorizer is None:
        st.error("⚠️ System Error: Models could not be found.")
        st.info("Troubleshooting: Please ensure 'model_class.pkl', 'model_score.pkl', and 'tfidf.pkl' are inside the 'models' folder on GitHub.")
    elif not desc_input:
        st.warning("Please enter at least a Problem Description.")
    else:
        # A. Preprocessing
        combined_text = f"{desc_input} {inp_input} {out_input}"
        
        # B. Feature Extraction
        text_vectorized = vectorizer.transform([combined_text])
        
        # C. Model Inference
        predicted_class = clf.predict(text_vectorized)[0]
        predicted_score = reg.predict(text_vectorized)[0]
        
        # --- 5. Display Results ---
        st.divider()
        st.subheader("Analysis Results")
        
        color_map = {"Easy": "green", "Medium": "orange", "Hard": "red"}
        color = color_map.get(predicted_class, "blue")
        
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.markdown("**Predicted Class**")
            st.markdown(f"<h1 style='color:{color}'>{predicted_class}</h1>", unsafe_allow_html=True)
            
        with result_col2:
            st.markdown("**Predicted Score (0-100)**")
            st.metric(label="Difficulty Score", value=f"{predicted_score:.1f}")
            st.progress(min(predicted_score / 100, 1.0))
            
        st.success("Prediction generated successfully.")
