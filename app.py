"""
Streamlit Frontend for LLM Evaluation Pipeline
Provides UI for uploading files and viewing evaluation results
"""

import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.express as px
from typing import Dict, List, Any, Optional

# API base URL
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="LLM Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 LLM Evaluation Pipeline Dashboard")
st.markdown("Evaluate RAG system responses with multi-level metrics")

# Initialize session state
if 'task_id' not in st.session_state:
    st.session_state.task_id = None
if 'immediate_results' not in st.session_state:
    st.session_state.immediate_results = None
if 'polling_active' not in st.session_state:
    st.session_state.polling_active = False
if 'final_results' not in st.session_state:
    st.session_state.final_results = None


def upload_files():
    """Handle file uploads."""
    st.header("📁 Upload Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        conversation_file = st.file_uploader(
            "Upload Conversation JSON",
            type=['json'],
            key='conversation'
        )
    
    with col2:
        context_vectors_file = st.file_uploader(
            "Upload Context Vectors JSON",
            type=['json'],
            key='context_vectors'
        )
    
    return conversation_file, context_vectors_file


def send_evaluation_request(conversation_json: Dict, context_vectors_json: Dict) -> Optional[Dict]:
    """Send evaluation request to API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/evaluate",
            json={
                "conversation_json": conversation_json,
                "context_vectors_json": context_vectors_json
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling API: {str(e)}")
        return None


def poll_task_status(task_id: str) -> Optional[Dict]:
    """Poll task status from API."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/results/{task_id}",
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error polling task status: {str(e)}")
        return None


def display_immediate_results(results: List[Dict[str, Any]]):
    """Display Level 1 and Retrieval metrics."""
    if not results:
        st.warning("No immediate results available.")
        return
    
    st.header("⚡ Immediate Results (Level 1 & Retrieval Metrics)")
    
    # Calculate averages
    level1_scores = []
    retrieval_scores = []
    
    for result in results:
        metrics = result.get('metrics', {})
        
        # Level 1 metrics
        level1 = metrics.get('level1', {})
        if level1:
            factuality = level1.get('factuality', {}).get('rouge_l_recall', 0)
            relevance = level1.get('relevance', {}).get('jaccard_similarity', 0)
            completeness = level1.get('completeness', {}).get('length_ratio', 0)
            level1_scores.append({
                'factuality': factuality,
                'relevance': relevance,
                'completeness': completeness
            })
        
        # Retrieval metrics
        retrieval = metrics.get('retrieval', {})
        if retrieval:
            retrieval_scores.append({
                'mrr': retrieval.get('mrr', 0),
                'ndcg': retrieval.get('ndcg', 0),
                'noise_ratio': retrieval.get('noise_ratio', 0),
                'hit_rate_at_3': retrieval.get('hit_rate_at_3', 0),
                'hit_rate_at_5': retrieval.get('hit_rate_at_5', 0),
                'hit_rate_at_10': retrieval.get('hit_rate_at_10', 0)
            })
    
    # Display metrics in columns
    if level1_scores:
        avg_factuality = sum(s['factuality'] for s in level1_scores) / len(level1_scores)
        avg_relevance = sum(s['relevance'] for s in level1_scores) / len(level1_scores)
        avg_completeness = sum(s['completeness'] for s in level1_scores) / len(level1_scores)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Factuality (ROUGE-L)", f"{avg_factuality:.3f}")
        with col2:
            st.metric("Average Relevance (Jaccard)", f"{avg_relevance:.3f}")
        with col3:
            st.metric("Average Completeness", f"{avg_completeness:.3f}")
    
    if retrieval_scores:
        avg_mrr = sum(s['mrr'] for s in retrieval_scores) / len(retrieval_scores)
        avg_ndcg = sum(s['ndcg'] for s in retrieval_scores) / len(retrieval_scores)
        avg_noise = sum(s['noise_ratio'] for s in retrieval_scores) / len(retrieval_scores)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average MRR", f"{avg_mrr:.3f}")
        with col2:
            st.metric("Average NDCG", f"{avg_ndcg:.3f}")
        with col3:
            st.metric("Average Noise Ratio", f"{avg_noise:.3f}")
    
    # Display detailed results table
    st.subheader("📋 Detailed Results")
    
    # Prepare data for table
    table_data = []
    for result in results:
        message_id = result.get('message_id', 'N/A')
        metrics = result.get('metrics', {})
        
        level1 = metrics.get('level1', {})
        retrieval = metrics.get('retrieval', {})
        
        row = {
            'Message ID': message_id,
            'Factuality': level1.get('factuality', {}).get('rouge_l_recall', 0),
            'Relevance': level1.get('relevance', {}).get('jaccard_similarity', 0),
            'Completeness': level1.get('completeness', {}).get('length_ratio', 0),
            'MRR': retrieval.get('mrr', 0),
            'NDCG': retrieval.get('ndcg', 0),
            'Noise Ratio': retrieval.get('noise_ratio', 0)
        }
        table_data.append(row)
    
    if table_data:
        df = pd.DataFrame(table_data)
        
        # Highlight rows with low factuality
        def highlight_low_factuality(row):
            if row['Factuality'] < 0.5:
                return ['background-color: #ffcccc'] * len(row)
            return [''] * len(row)
        
        styled_df = df.style.apply(highlight_low_factuality, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # Visualization: Used vs Retrieved Vectors
        if retrieval_scores:
            st.subheader("📊 Retrieval Statistics")
            
            # Calculate average vectors used vs retrieved
            # This is a simplified visualization
            # In a real scenario, you'd track this per message
            avg_hit_rate_3 = sum(s['hit_rate_at_3'] for s in retrieval_scores) / len(retrieval_scores)
            avg_hit_rate_5 = sum(s['hit_rate_at_5'] for s in retrieval_scores) / len(retrieval_scores)
            avg_hit_rate_10 = sum(s['hit_rate_at_10'] for s in retrieval_scores) / len(retrieval_scores)
            
            hit_rate_data = pd.DataFrame({
                'K': [3, 5, 10],
                'Hit Rate': [avg_hit_rate_3, avg_hit_rate_5, avg_hit_rate_10]
            })
            
            fig = px.bar(
                hit_rate_data,
                x='K',
                y='Hit Rate',
                title='Average Hit Rate @ K',
                labels={'K': 'Top K Results', 'Hit Rate': 'Hit Rate'}
            )
            st.plotly_chart(fig, use_container_width=True)


def display_async_results(level2_results: List[Dict], level3_results: List[Dict]):
    """Display Level 2 and Level 3 metrics."""
    st.header("🔄 Semantic & LLM Metrics (Level 2 & 3)")
    
    if level2_results:
        st.subheader("📊 Level 2: Semantic Metrics")
        
        level2_data = []
        for result in level2_results:
            message_id = result.get('message_id', 'N/A')
            metrics = result.get('metrics', {}).get('level2', {})
            
            if metrics:
                level2_data.append({
                    'Message ID': message_id,
                    'Factuality (NLI)': metrics.get('factuality', {}).get('entailment_score', 0),
                    'Relevance (SBERT)': metrics.get('relevance', {}).get('semantic_similarity', 0),
                    'Completeness (IFEval)': metrics.get('completeness', {}).get('instruction_score', 0)
                })
        
        if level2_data:
            df_level2 = pd.DataFrame(level2_data)
            st.dataframe(df_level2, use_container_width=True)
    
    if level3_results:
        st.subheader("🤖 Level 3: LLM-Based Metrics")
        
        level3_data = []
        for result in level3_results:
            message_id = result.get('message_id', 'N/A')
            metrics = result.get('metrics', {}).get('level3', {})
            
            if metrics:
                level3_data.append({
                    'Message ID': message_id,
                    'Factuality (FActScore)': metrics.get('factuality', {}).get('factscore', 0),
                    'Relevance (Ragas)': metrics.get('relevance', {}).get('ragas_relevance', 0),
                    'Completeness (REC)': metrics.get('completeness', {}).get('rec_score', 0)
                })
        
        if level3_data:
            df_level3 = pd.DataFrame(level3_data)
            st.dataframe(df_level3, use_container_width=True)
    
    if not level2_results and not level3_results:
        st.info("No Level 2 or Level 3 results available yet. These metrics are sampled (10% and 1% respectively).")


def main():
    """Main Streamlit app."""
    # Sidebar for file upload
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("Upload your JSON files to start evaluation")
        
        conversation_file, context_vectors_file = upload_files()
        
        if st.button("🚀 Start Evaluation", type="primary", use_container_width=True):
            if conversation_file is None or context_vectors_file is None:
                st.error("Please upload both files before starting evaluation.")
            else:
                try:
                    # Parse JSON files
                    conversation_json = json.load(conversation_file)
                    context_vectors_json = json.load(context_vectors_file)
                    
                    # Show loading spinner
                    with st.spinner("Sending evaluation request..."):
                        response = send_evaluation_request(conversation_json, context_vectors_json)
                    
                    if response:
                        st.session_state.task_id = response.get('task_id')
                        st.session_state.immediate_results = response.get('immediate_results', [])
                        st.session_state.polling_active = True
                        st.success("Evaluation started! Immediate results available.")
                        st.rerun()
                
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON format: {str(e)}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Reset button
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.task_id = None
            st.session_state.immediate_results = None
            st.session_state.polling_active = False
            st.session_state.final_results = None
            st.rerun()
    
    # Main content area
    if st.session_state.immediate_results:
        # Display immediate results
        display_immediate_results(st.session_state.immediate_results)
        
        # Poll for async results if task_id exists
        if st.session_state.task_id and st.session_state.polling_active:
            st.divider()
            
            status_response = poll_task_status(st.session_state.task_id)
            
            if status_response:
                status = status_response.get('status', 'processing')
                
                if status == 'completed':
                    st.session_state.polling_active = False
                    st.session_state.final_results = status_response
                    st.success("✅ All metrics calculated!")
                elif status == 'error':
                    st.error(f"Error in background processing: {status_response.get('error', 'Unknown error')}")
                    st.session_state.polling_active = False
                else:
                    # Still processing - show status and auto-refresh
                    status_placeholder = st.empty()
                    with status_placeholder:
                        st.info("🔄 Calculating Semantic & LLM Metrics... This may take a few moments.")
                    # Auto-refresh after 3 seconds
                    time.sleep(3)
                    st.rerun()
        
        # Display async results if available
        if st.session_state.final_results:
            level2_results = st.session_state.final_results.get('level2_results', [])
            level3_results = st.session_state.final_results.get('level3_results', [])
            display_async_results(level2_results, level3_results)
    
    else:
        # Welcome screen
        st.info("👈 Please upload conversation.json and context_vectors.json files in the sidebar to begin evaluation.")
        
        st.markdown("""
        ### 📋 What this dashboard does:
        
        1. **Immediate Results**: Level 1 (heuristic) and Retrieval metrics are calculated instantly
        2. **Background Processing**: Level 2 (semantic) and Level 3 (LLM-based) metrics are calculated asynchronously
        3. **Real-time Updates**: The dashboard automatically polls for completion
        
        ### 📊 Metrics Explained:
        
        - **Level 1**: Fast heuristic metrics (ROUGE-L, Jaccard, Length Ratio)
        - **Retrieval**: Ranking metrics (MRR, NDCG, Hit Rate @ K)
        - **Level 2**: Semantic metrics using BERT/SBERT models
        - **Level 3**: LLM-based robust metrics (FActScore, Ragas, REC)
        """)


if __name__ == "__main__":
    main()

