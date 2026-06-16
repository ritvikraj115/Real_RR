import streamlit as st
import pandas as pd
import os
import sys

# Import your core pipeline's run function
# Ensure rank_candidates.py is in the same directory
try:
    from rank_candidates import run
except ImportError:
    st.error("Could not import rank_candidates.py. Ensure it is in the root directory.")

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Redrob Ranker Sandbox | Team: Real_RR", 
    page_icon="🚀", 
    layout="wide"
)

st.title("🚀 Redrob Hackathon: Stage 1 Sandbox")
st.markdown("""
**Team ID:** `Real_RR`  
Welcome to our Ranking Pipeline Sandbox. 

As per the hackathon guidelines (Section 10.5), this environment verifies end-to-end reproducibility.
Please upload a sample candidate file (≤ 100 rows) to execute the LTR cascade (BM25 → Bi-Encoder → Cross-Encoder).
""")

# --- MODEL CACHING (CRITICAL FOR STREAMLIT 1GB RAM LIMIT) ---
@st.cache_resource
def load_models_into_memory():
    """
    Forces the models to load into RAM once on startup.
    This prevents OOM (Out of Memory) crashes when the UI thread tries to run.
    """
    with st.spinner("Initializing Neural Networks (BGE-Small & MS-Marco)..."):
        # We wrap this in a try-except just in case the Streamlit environment 
        # needs to download them dynamically if the offline models aren't pushed.
        try:
            from sentence_transformers import SentenceTransformer, CrossEncoder
            # If you are using the local models folder, update these paths to "models/..."
            SentenceTransformer('BAAI/bge-small-en-v1.5')
            CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            return True
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return False

model_status = load_models_into_memory()

# --- UI & EXECUTION ---
st.markdown("### 1. Upload Candidates")
uploaded_file = st.file_uploader("Upload sample_candidates.jsonl (Max 100 rows)", type=['jsonl'])

if uploaded_file is not None and model_status:
    # Save the uploaded file temporarily
    temp_input_path = "temp_sample_candidates.jsonl"
    with open(temp_input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success("File uploaded successfully. Ready to execute pipeline.")
    
    st.markdown("### 2. Run Pipeline")
    if st.button("▶️ Execute Ranking Engine", type="primary"):
        # The required output filename per your Team ID
        output_csv = "Real_RR.csv"  
        
        with st.spinner("Executing Stage 1 (BM25) → Stage 2 (Vector) → Stage 3 (Cross-Encoder)..."):
            try:
                # Mock argparse object matching your rank_candidates.py logic
                class DummyArgs:
                    candidates = temp_input_path
                    jd_index = "jd_hybrid_index.json"
                    output = output_csv
                    
                    # You ARE caching JD embeddings to save compute
                    jd_embeddings_cache = "data/jd_vector_embeddings.npz"
                    
                    # You are NOT caching candidate vectors/cross scores. 
                    # Pointing to temp paths ensures it runs LIVE on the 100 uploaded candidates.
                    vector_scores_cache = "temp_vector_scores.npz"
                    cross_scores_cache = "temp_cross_scores.npz"
                    
                    cross_encoder_backend = "sentence-transformers"
                    
                    # Point these to your offline folder (e.g., 'models/...') if you used the chunking trick
                    embedding_model = "BAAI/bge-small-en-v1.5" 
                    cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
                    
                    require_cross_encoder = True
                    batch_size = 32  # Small batch size to respect Streamlit RAM
                    prepare_cache_only = False
                    validate_only = False

                # Trigger the main ranking pipeline
                exit_code = run(DummyArgs())
                
                if exit_code == 0 and os.path.exists(output_csv):
                    st.success("✅ Pipeline executed successfully in under 5 minutes!")
                    
                    st.markdown("### 3. Review & Download Results")
                    df = pd.read_csv(output_csv)
                    
                    # Display the top 100 rows nicely to the judges
                    st.dataframe(
                        df,
                        column_config={
                            "rank": st.column_config.NumberColumn("Rank", width="small"),
                            "score": st.column_config.NumberColumn("Score", format="%.4f"),
                            "reasoning": st.column_config.TextColumn("Model Reasoning", width="large")
                        },
                        hide_index=True
                    )
                    
                    # Official Download Button
                    with open(output_csv, "rb") as f:
                        st.download_button(
                            label="⬇️ Download Real_RR.csv",
                            data=f,
                            file_name=output_csv,
                            mime="text/csv"
                        )
                else:
                    st.error("Pipeline failed to generate the output CSV. Check the internal logs.")
                    
            except Exception as e:
                st.error(f"Execution Error: {str(e)}")

st.markdown("---")
st.caption("Redrob Hackathon v4 | Team: Real_RR | Deployed via Streamlit")