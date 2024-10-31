import streamlit as st
import PyPDF2
import pandas as pd
import json
import io
from typing import List, Tuple, Dict
import tempfile
from sklearn.model_selection import train_test_split
import requests
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize session state variables if they don't exist
if 'train_df' not in st.session_state:
    st.session_state.train_df = None
if 'val_df' not in st.session_state:
    st.session_state.val_df = None
if 'generated' not in st.session_state:
    st.session_state.generated = False
if 'previous_upload_state' not in st.session_state:
    st.session_state.previous_upload_state = False

def reset_session_state():
    """Reset all relevant session state variables"""
    st.session_state.train_df = None
    st.session_state.val_df = None
    st.session_state.generated = False

def parse_pdf(uploaded_file) -> str:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file.seek(0)
        
        reader = PyPDF2.PdfReader(tmp_file.name)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def generate_qa_pairs(text: str, api_key: str, model: str, num_pairs: int, context: str) -> pd.DataFrame:
    url = "https://generativelanguage.googleapis.com/v1beta/cachedContents:query"  # Replace with the actual Gemini API endpoint
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "prompt": f"""
        Given the following text, generate {num_pairs} question-answer pairs:

        {text}

        Format each pair as:
        Q: [Question]
        A: [Answer]

        Ensure the questions are diverse and cover different aspects of the text.
        """
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        qa_text = response.json()['qa_pairs']
        qa_pairs = []
        
        for pair in qa_text.split('\n\n'):
            if pair.startswith('Q:') and 'A:' in pair:
                question, answer = pair.split('A:')
                question = question.replace('Q:', '').strip()
                answer = answer.strip()
                qa_pairs.append({
                    'Question': question,
                    'Answer': answer,
                    'Context': context
                })

        return pd.DataFrame(qa_pairs)
    
    except Exception as e:
        st.error(f"Error generating QA pairs: {str(e)}")
        return pd.DataFrame()

def create_jsonl_content(df: pd.DataFrame, system_content: str) -> str:
    """Convert DataFrame to JSONL string content"""
    jsonl_content = []
    for _, row in df.iterrows():
        entry = {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": row['Question']},
                {"role": "assistant", "content": row['Answer']}
            ]
        }
        jsonl_content.append(json.dumps(entry, ensure_ascii=False))
    return '\n'.join(jsonl_content)

def process_and_split_data(text: str, api_key: str, model: str, num_pairs: int, context: str, train_size: float):
    """Process data and store results in session state"""
    df = generate_qa_pairs(text, api_key, model, num_pairs, context)
    
    if not df.empty:
        # Split the dataset
        train_df, val_df = train_test_split(
            df, 
            train_size=train_size/100,
            random_state=42
        )
        
        # Store in session state
        st.session_state.train_df = train_df
        st.session_state.val_df = val_df
        st.session_state.generated = True
        return True
    return False

def main():
    st.title("LLM Dataset Generator")
    st.write("Upload a PDF file and generate training & validation sets of question-answer pairs of your data using LLM.")

    # Sidebar configurations
    st.sidebar.header("Configuration")
    
    model = "gemini-1.5-pro"
    # model = st.sidebar.selectbox(
    #     "Select Model",
    #     ["gemini-1", "gemini-2", "gemini-3"]
    # )
    
    num_pairs = st.sidebar.number_input(
        "Number of QA Pairs",
        min_value=1,
        max_value=10000,
        value=5
    )
    
    context = st.sidebar.text_area(
        "Custom Context",
        value="Write a response that appropriately completes the request.",
        help="This text will be added to the Context column for each QA pair.",
        placeholder= "Add custom context here."
    )

    # Dataset split configuration
    st.sidebar.header("Dataset Split")
    train_size = st.sidebar.slider(
        "Training Set Size (%)",
        min_value=50,
        max_value=90,
        value=80,
        step=5
    )

    # Output format configuration
    st.sidebar.header("Output Format")
    output_format = st.sidebar.selectbox(
        "Select Output Format",
        ["CSV", "JSONL"]
    )

    if output_format == "JSONL":
        system_content = st.sidebar.text_area(
            "System Message",
            value="You are a helpful assistant that provides accurate and informative answers.",
            help="This message will be used as the system content in the JSONL format."
        )

    # Main area
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    # Check if upload state has changed
    current_upload_state = uploaded_file is not None
    if current_upload_state != st.session_state.previous_upload_state:
        if not current_upload_state:  # File was removed
            reset_session_state()
        st.session_state.previous_upload_state = current_upload_state

    if uploaded_file is not None:
        text = parse_pdf(uploaded_file)
        st.success("PDF processed successfully!")
        
        if st.button("Generate QA Pairs"):
            with st.spinner("Generating QA pairs..."):
                success = process_and_split_data(text, api_key, model, num_pairs, context, train_size)
                if success:
                    st.success("QA pairs generated successfully!")

    # Display results if data has been generated
    if st.session_state.generated and st.session_state.train_df is not None and st.session_state.val_df is not None:
        # Display the dataframes
        st.subheader("Training Set")
        st.dataframe(st.session_state.train_df)
        
        st.subheader("Validation Set")
        st.dataframe(st.session_state.val_df)
        
        # Create download section
        st.subheader("Download Generated Datasets")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Training Set")
            if output_format == "CSV":
                train_csv = st.session_state.train_df.to_csv(index=False)
                st.download_button(
                    label="Download Training Set (CSV)",
                    data=train_csv,
                    file_name="train_qa_pairs.csv",
                    mime="text/csv",
                    key="train_csv"
                )
            else:  # JSONL format
                train_jsonl = create_jsonl_content(st.session_state.train_df, system_content)
                st.download_button(
                    label="Download Training Set (JSONL)",
                    data=train_jsonl,
                    file_name="train_qa_pairs.jsonl",
                    mime="application/jsonl",
                    key="train_jsonl"
                )
        
        with col2:
            st.markdown("##### Validation Set")
            if output_format == "CSV":
                val_csv = st.session_state.val_df.to_csv(index=False)
                st.download_button(
                    label="Download Validation Set (CSV)",
                    data=val_csv,
                    file_name="val_qa_pairs.csv",
                    mime="text/csv",
                    key="val_csv"
                )
            else:  # JSONL format
                val_jsonl = create_jsonl_content(st.session_state.val_df, system_content)
                st.download_button(
                    label="Download Validation Set (JSONL)",
                    data=val_jsonl,
                    file_name="val_qa_pairs.jsonl",
                    mime="application/jsonl",
                    key="val_jsonl"
                )
        
        # Display statistics
        st.subheader("Statistics")
        st.write(f"Total QA pairs: {len(st.session_state.train_df) + len(st.session_state.val_df)}")
        st.write(f"Training set size: {len(st.session_state.train_df)} ({train_size}%)")
        st.write(f"Validation set size: {len(st.session_state.val_df)} ({100-train_size}%)")
        st.write(f"Average question length: {st.session_state.train_df['Question'].str.len().mean():.1f} characters")
        st.write(f"Average answer length: {st.session_state.train_df['Answer'].str.len().mean():.1f} characters")

if __name__ == "__main__":
    st.set_page_config(
        page_title="LLM Dataset Generator",
        page_icon="📚",
        layout="wide"
    )
    main()