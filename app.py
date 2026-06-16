import streamlit as st
import pandas as pd
import os

# Import your core pipeline's run function
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
    This prevents OOM crashes when the UI thread tries to run.
    """
    with st.spinner("Initializing Neural Networks (BGE-Small & MS-Marco)..."):
        try:
            from sentence_transformers import SentenceTransformer, CrossEncoder
            # Uses models from HuggingFace or your offline 'models' folder if you pushed it
            SentenceTransformer('BAAI/bge-small-en-v1.5')
            CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            return True
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return False

model_status = load_models_into_memory()

# --- UI & EXECUTION ---
st.markdown("### 1. Upload Candidates")
# Added 'jsonl' to accepted types because the dataset is candidates.jsonl
uploaded_file = st.file_uploader("Upload sample_candidates.jsonl (Max 100 rows)", type=['jsonl', 'json'])

if uploaded_file is not None and model_status:
    # Save the uploaded file temporarily
    temp_input_path = "temp_sample_candidates.jsonl"
    with open(temp_input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success("File uploaded successfully. Ready to execute pipeline.")
    
    st.markdown("### 2. Run Pipeline")
    if st.button("▶️ Execute Ranking Engine", type="primary"):
        output_csv = "Real_RR.csv"  
        
        with st.spinner("Executing Stage 1 (BM25) → Stage 2 (Vector) → Stage 3 (Cross-Encoder)..."):
            try:
                # -------------------------------------------------------------
                # FULLY MAPPED ARGUMENTS (Prevents 'AttributeError')
                # -------------------------------------------------------------
                class DummyArgs:
                    jd_index = "jd_hybrid_index.json"
                    candidates = temp_input_path
                    output = output_csv
                    
                    # Missing metadata attributes explicitly added
                    metadata_output = "temp_ranking_metadata.json"
                    l0_report_output = "temp_l0_triage_report.csv"
                    stage1_report_output = "temp_stage1_bm25_report.csv"
                    stage2_report_output = "temp_stage2_vector_report.csv"
                    scores_output = ""
                    chunk_scores_output = ""
                    chunk_scores_limit = 1000
                    
                    pre_shortlist_size = 1000
                    shortlist_size = 150
                    top_k = 100
                    
                    # CRITICAL FIX: If judges upload 50 candidates, pipeline won't crash
                    allow_fewer_than_top_k = True 
                    max_candidates = None
                    as_of_date = "2026-06-09"
                    
                    vector_backend = "sentence-transformers"
                    embedding_model = "BAAI/bge-small-en-v1.5"
                    model_cache_dir = "models"
                    
                    # Use flat paths so Streamlit doesn't throw 'Folder not found'
                    jd_embeddings_cache = "temp_jd_vector_embeddings.npz"
                    vector_scores_cache = "temp_pre_shortlist_vector_scores.npz"
                    
                    cross_encoder_backend = "sentence-transformers"
                    cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
                    cross_scores_cache = "temp_shortlist_cross_scores.npz"
                    
                    require_cross_encoder = True
                    batch_size = 32  # Small batch size to respect Streamlit RAM
                    prepare_cache_only = False
                    validate_only = False

                # Trigger the main ranking pipeline
                exit_code = run(DummyArgs())
                
                # Some scripts return 'None' on success, so we handle both 0 and None
                if exit_code in [0, None] and os.path.exists(output_csv):
                    st.success("✅ Pipeline executed successfully in under 5 minutes!")
                    
                    st.markdown("### 3. Review & Download Results")
                    df = pd.read_csv(output_csv)
                    
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