import streamlit as st
import joblib
import os

# --- 1. Load the Trained AI Models ---
# We use @st.cache_resource so it only loads once (faster)
@st.cache_resource
def load_models():
    try:
        # Get the current directory where app.py is located
        base_dir = os.path.dirname(__file__)
        
        # Construct specific paths to the 'models' folder
        path_clf = os.path.join(base_dir, 'models', 'model_class.pkl')
        path_reg = os.path.join(base_dir, 'models', 'model_score.pkl')
        path_vec = os.path.join(base_dir, 'models', 'tfidf.pkl')

        # Load the models
        clf = joblib.load(path_clf)
        reg = joblib.load(path_reg)
        vectorizer = joblib.load(path_vec)
        
        return clf, reg, vectorizer
    except Exception as e:
        st.error(f"Error loading models. Please ensure the 'models/' folder exists. Details: {e}")
        return None, None, None

clf, reg, vectorizer = load_models()

# --- 2. Configure the Page ---
st.set_page_config(page_title="AutoJudge AI", page_icon="🤖", layout="centered")

st.title("🤖 AutoJudge: Difficulty Predictor")
st.markdown("### Professional AI System")
st.markdown("Enter the problem details below to predict its difficulty using Machine Learning.")

# --- 3. The 3 Input Fields ---
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
    if not clf:
        st.error("Models failed to load. Check your 'models' folder.")
    elif not desc_input:
        st.warning("Please enter at least a Problem Description.")
    else:
        # A. Preprocessing: Combine all text inputs
        combined_text = f"{desc_input} {inp_input} {out_input}"
        
        # B. Feature Extraction: Convert text to numbers (TF-IDF)
        text_vectorized = vectorizer.transform([combined_text])
        
        # C. Model Inference
        predicted_class = clf.predict(text_vectorized)[0]  # Easy / Medium / Hard
        predicted_score = reg.predict(text_vectorized)[0]  # Numerical Value
        
        # --- 5. Display Results ---
        st.divider()
        st.subheader("Analysis Results")
        
        # Set colors for visual appeal
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
            
        st.success("Prediction generated successfully using Random Forest models.")
