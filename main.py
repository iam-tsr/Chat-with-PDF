import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import subprocess
import shutil
import mimetypes

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Create downloads directory if it doesn't exist
DOWNLOADS_DIR = "downloads"
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def convert_to_pdf(input_file):
    output_file = os.path.join(DOWNLOADS_DIR, os.path.splitext(os.path.basename(input_file))[0] + '.pdf')
    subprocess.run(['unoconv', '-f', 'pdf', '-o', output_file, input_file])
    return output_file

def is_pdf(file):
    # Check if file is PDF based on mime type and extension
    mime_type, _ = mimetypes.guess_type(file.name)
    return mime_type == 'application/pdf' or file.name.lower().endswith('.pdf')

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDFs")
    
    # Initialize session state for tracking conversions
    if 'converted_files' not in st.session_state:
        st.session_state.converted_files = {}
    
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)
    
    with st.sidebar:
        st.title("Menu:")
        uploaded_file = st.file_uploader("Upload your File", accept_multiple_files=False)
        
        if uploaded_file:
            # Separate PDF and non-PDF files
            pdf_files = []
            non_pdf_files = []
            if is_pdf(uploaded_file):
                pdf_files.append(uploaded_file)
            else:
                non_pdf_files.append(uploaded_file)
            
            # Handle non-PDF files first
            if non_pdf_files:
                st.warning(f"Found {len(non_pdf_files)} non-PDF files that need conversion:")
                
                # Convert non-PDF files
                for file in non_pdf_files:
                    if file.name not in st.session_state.converted_files:
                        # Save and convert file
                        temp_path = os.path.join(DOWNLOADS_DIR, file.name)
                        with open(temp_path, "wb") as f:
                            f.write(file.getbuffer())
                        
                        pdf_path = convert_to_pdf(temp_path)
                        st.session_state.converted_files[file.name] = pdf_path
                        os.remove(temp_path)  # Clean up original file
                
                # Display download section for converted files
                st.subheader("Download Converted PDFs:")
                for original_name, pdf_path in st.session_state.converted_files.items():
                    try:
                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                label=f"📥 Download {original_name} as PDF",
                                data=f.read(),
                                file_name=os.path.basename(pdf_path),
                                mime="application/pdf",
                                key=f"download_{original_name}"
                            )
                    except FileNotFoundError:
                        st.error(f"Error: Could not find converted file for {original_name}")
                
                st.error("⚠️ Please download the converted PDFs and reupload them before processing")
                return  # Stop here and don't show the Process button
        
        # Only show Process button if all files are PDFs
        if uploaded_file and is_pdf(uploaded_file):
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    temp_path = os.path.join(DOWNLOADS_DIR, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process the PDF
                    raw_text = get_pdf_text([temp_path])
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    
                    # Clean up
                    os.remove(temp_path)
                    
                    st.success(f"✅ Successfully processed the PDF file")
                    
                    # Clear conversion history after successful processing
                    st.session_state.converted_files = {}

if __name__ == "__main__":
    main()