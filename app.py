import streamlit as st
import io
import PyPDF2
import joblib
import os

# --- 1. Load Trained Models & Vectorizer Safely ---
@st.cache_resource
def load_ml_model(model_path):
    if os.path.exists(model_path):
        loaded_object = joblib.load(model_path)
        # Extra safety check: ensuring we don't accidentally return a string path
        if isinstance(loaded_object, str):
            return None
        return loaded_object
    return None

@st.cache_resource
def load_vectorizer():
    if os.path.exists("tfidf_vectorizer.pkl"):
        return joblib.load("tfidf_vectorizer.pkl")
    return None

tfidf_vectorizer = load_vectorizer()

# --- 2. Clean Text Helper Function (Warning Fixed with raw string r""") ---
def clean_text(text):
    import re
    text = text.lower()
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', '  ', text)
    # Fixed syntax warning by removing redundant escape setups
    text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\\]^_`{|}~]', ' ', text)
    text = re.sub(r'[^\x00-\x7f]', r' ', text) 
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# --- 3. UI Layout Configuration ---
st.set_page_config(page_title="Resume Screening NLP", page_icon="📊", layout="centered")

st.title("📄 AI Resume Screening Assistant")
st.subheader("Evaluate candidate alignment using custom ML architectures")

# Default Job Requirements
default_job_req = """
Programming: Python, SQL
Machine Learning: scikit-learn, SVM, Random Forest, Logistic Regression, K-Means, Decision Trees
Data Analysis & Visualization: NumPy, Pandas, Matplotlib, Seaborn
Data Preprocessing: Feature Engineering, Data Cleaning, Handling Missing Values
Model Evaluation: Accuracy, Precision, Recall, F1-score
Tools: Git, GitHub, Jupyter Notebook, Google Colab
Soft Skills: Problem Solving, Analytical Thinking, Communication
"""

# Editable Job Description UI
with st.expander("💼 View/Edit Job Requirements", expanded=False):
    job_req_input = st.text_area("Target Profile Requirements:", value=default_job_req.strip(), height=180)

# --- 4. Model Selection Sidebar ---
st.sidebar.header("⚙️ Model Configuration")

model_options = {
    "Best Model (Default)": "best_model.pkl",
    "LightGBM": "lightgbm.pkl",
    "XGBoost": "xgboost.pkl",
    "Random Forest": "random_forest.pkl",
    "SVM": "svm.pkl",
    "Logistic Regression": "log_reg.pkl"
}

selected_model_name = st.sidebar.selectbox("Choose ML Model for Inference:", list(model_options.keys()))
model_file = model_options[selected_model_name]

# Explicitly fetch the unpacked object here
selected_model = load_ml_model(model_file)

if selected_model is None:
    st.sidebar.error(f"⚠️ `{model_file}` could not be loaded or is missing.")
else:
    st.sidebar.success(f"🤖 Active Model: {selected_model_name}")

# --- 5. File Processing & Upload ---
uploaded_file = st.file_uploader("Upload candidate resume (PDF or TXT format)", type=["pdf", "txt"])

if uploaded_file is not None:
    resume_text = ""
    filename = uploaded_file.name
    
    if filename.endswith(".txt"):
        resume_text = uploaded_file.read().decode("utf-8")
    elif filename.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                resume_text += content

    if resume_text.strip() == "":
        st.error("Empty text extracted. Ensure the file is not corrupted.")
    elif tfidf_vectorizer is None:
        st.error("Missing critical asset: `tfidf_vectorizer.pkl` not found in directory.")
    elif selected_model is None or isinstance(selected_model, str):
        st.error("Selected engine is not a valid ML model object.")
    else:
        st.info(f"Loaded: **{filename}**")
        
        # --- 6. Trigger Prediction ---
        if st.button("🚀 Analyze Resume Fit"):
            with st.spinner(f"Processing text via {selected_model_name}..."):
                
                # Combine and process text
                combined_text = job_req_input + " " + resume_text
                combined_clean = clean_text(combined_text)
                
                # NLP Vectorization
                combined_tfidf = tfidf_vectorizer.transform([combined_clean])
                
                # Dynamic prediction assignment depending on model requirements
                if "xgboost" in model_file or "lightgbm" in model_file:
                    prediction = selected_model.predict(combined_tfidf)[0] + 1
                else:
                    prediction = selected_model.predict(combined_tfidf)[0]
                
                # Output Results Visualizations
                st.markdown("---")
                st.markdown("### 📊 Inference Diagnostic Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Assigned Category Label ID", value=int(prediction))
                with col2:
                    st.metric(label="Pipeline Engine Used", value=selected_model_name)
                    
                st.balloons()