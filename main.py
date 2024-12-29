import os
import shutil
from dotenv import load_dotenv

import streamlit as st
from PyPDF2 import PdfReader

import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Delete existing local files (only once at the beginning)
if "delete_done" not in st.session_state:
    def delete_local(folder_path):
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
    delete_local("faiss_index")
    st.session_state["delete_done"] = True

# Initialize conversation history
conversation_history = []

# PDF text extraction function
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Create or load vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Retrieve top-k most relevant chunks
def get_relevant_context(user_question, top_k=5):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=top_k)  # Retrieve top-k relevant chunks
    return docs

# Format context from retrieved chunks
def format_context(docs):
    return " ".join([doc.page_content for doc in docs])

# Manage conversation history with a sliding window approach
def manage_conversation_history(user_question, bot_response, max_turns=3):
    global conversation_history
    if len(conversation_history) >= max_turns * 2:
        conversation_history = conversation_history[2:]  # Remove oldest interaction
    conversation_history.extend([f"User: {user_question}", f"Bot: {bot_response}"])
    return "\n".join(conversation_history)

# Define a prompt template with structured instructions
def get_conversational_chain():
    prompt_template = """
    You are an intelligent assistant. Read the context carefully, and answer as if you're explaining to a person in a simple and friendly manner.
    Use plain language, avoid technical jargon, and provide a concise response. Only structure your answer in points if there are multiple distinct points or steps to explain.

    <context>
    {context}
    </context>
    Question: {question}

    Your answer should be:
    - Friendly and conversational
    - In complete sentences if the answer is short or straightforward
    - Structured as a list only if there are multiple points or steps
    - Accurate and based only on the context provided
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.2)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Handle user input and generate response
def user_input(user_question):
    # Simplify the question to improve comprehension
    simplified_question = simplify_question(user_question)
    
    # Retrieve relevant context from FAISS
    docs = get_relevant_context(simplified_question, top_k=5)
    context = format_context(docs)
    
    # Generate response using the conversational chain
    chain = get_conversational_chain()
    response = chain.invoke({"input_documents": docs, "question": simplified_question}, return_only_outputs=True)
    
    # Manage conversation history
    bot_response = response["output_text"]
    conversation_history_text = manage_conversation_history(simplified_question, bot_response)
    
    # Display the response
    st.write(bot_response)
# TSR
def simplify_question(question):
    # Basic preprocessing steps
    question = question.lower().strip()  # Lowercase and trim whitespace
    question = question.replace("please", "").replace("could you", "").replace("?", "")
    # More advanced NLP techniques could be applied here, such as paraphrasing
    return question

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with PDFs")
    
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, type=["pdf"])
        
        if st.button("Submit & Process"):
            
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

        st.info("This app allows you to chat with PDF multiple files")

    if not pdf_docs:
        st.markdown("<h4>Instructions<", unsafe_allow_html=True)

        st.write("1. Upload your PDF files")
        st.write("2. Ask a question")
        st.write("3. Get an answer")

    else:
        user_question = st.text_input("Ask a Question from the PDF Files")
        if st.button("Ask"):
            user_input(user_question)

if __name__ == "__main__":
    main()
# TSR