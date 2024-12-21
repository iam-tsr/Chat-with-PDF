import os
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceInstructEmbeddings  
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# Initialize conversation history
conversation_history = []

load_dotenv()  # Load API key from .env file
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Function to call Gemini API for text generation
def call_gemini_api(prompt, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")  # Specify the Gemini model
    response = model.generate_content(prompt)  

    # Check if response is valid and return the generated content
    if response:
        return response.text
    else:
        return "I don't know."

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
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Retrieve top-k most relevant chunks
def get_relevant_context(user_question, top_k=5):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question, k=top_k)  

# Format context from retrieved chunks
def format_context(docs):
    return " ".join([doc.page_content for doc in docs])

# Manage conversation history with a sliding window approach
def manage_conversation_history(user_question, bot_response, max_turns=3):
    global conversation_history
    if len(conversation_history) >= max_turns * 2:
        conversation_history = conversation_history[2:]  
    conversation_history.extend([f"User: {user_question}", f"Bot: {bot_response}"])
    return "\n".join(conversation_history)

# Define the prompt template
def get_conversational_chain(api_key):
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

    # Create the RetrievalQA chain using the function to call Gemini API
    def qa_chain(query, context):
        prompt = prompt_template.format(context=context, question=query)
        return call_gemini_api(prompt, api_key)

    return qa_chain

# Handle user input and generate response
def user_input(user_question, api_key):
    # Simplify the question to improve comprehension
    simplified_question = simplify_question(user_question)

    # Retrieve relevant context from FAISS
    docs = get_relevant_context(simplified_question, top_k=5)
    context = format_context(docs)

    # Generate response using the conversational chain
    chain = get_conversational_chain(api_key)
    response = chain(simplified_question, context)

    # Manage conversation history
    bot_response = response
    conversation_history_text = manage_conversation_history(simplified_question, bot_response)

    # Display the response
    st.write(bot_response)

# TSR - Simplify the question
def simplify_question(question):
    # Basic preprocessing steps
    question = question.lower().strip()  # Lowercase and trim whitespace
    question = question.replace("please", "").replace("could you", "").replace("?", "")
    return question

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="TSR")
    st.header("Chat with PDF")
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question, gemini_api_key)

    with st.sidebar:
        st.title("Menu:")

        api_key = st.text_input("Enter your Gemini API key:", type="password")
        if st.button("Submit API Key"):
            if api_key:
                # Store the API key in Streamlit's session state for later use
                st.session_state['gemini_api_key'] = api_key
                st.success("API key saved successfully!")
            else:
                st.error("Please enter a valid API key.")

        if 'gemini_api_key' in st.session_state:
            pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type="pdf")
            if st.button("Submit & Process PDFs"):
                if pdf_docs:
                    with st.spinner("Processing PDFs..."):
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("PDFs processed successfully!")
                else:
                    st.error("Please upload PDF files to process.")

if __name__ == "__main__":
    main()
