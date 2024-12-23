import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
import streamlit as st
import os
from dotenv import load_dotenv


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
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Retrieve top-k most relevant chunks
def get_relevant_context(user_question, top_k=5):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
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
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)
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
    response = chain({"input_documents": docs, "question": simplified_question}, return_only_outputs=True)
    
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
    return question

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="TSR")
    st.header("Chat with PDF")
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)
    
    with st.sidebar:
        st.title("Menu:")

        # ---In-case you want to use the Google API Key from the .env file---
        # load_dotenv()
        # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

        api_key = st.text_input("Enter your Gemini API key:", type="password")
        if st.button("Submit API Key"):
            if api_key:
                # Store the API key in Streamlit's session state for later use
                st.session_state['gemini_api_key'] = api_key
                genai.configure(api_key=api_key)
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
# TSR