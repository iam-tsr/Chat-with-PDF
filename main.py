import streamlit as st
from dotenv import load_dotenv
import os
from file_extractor import load_dataset

load_dotenv()

# Load your dataset
dataset_content = load_dataset("VONG.pdf")

# Debug: Print dataset content length
print(f"Dataset content length: {len(dataset_content)}")

# Define your keywords
keywords = ["vong", "keyword2", "keyword3"]  # Add more keywords as needed

# Function to check if input is relevant
def is_relevant_input(input, keywords):
    for keyword in keywords:
        if keyword.lower() in input.lower():
            return True
    return False

# Function to handle input and generate response
def handle_input(input):
    # Check if input matches any keyword
    if is_relevant_input(input, keywords):
        # Extract relevant part from dataset content
        start_index = dataset_content.lower().find(input.lower())
        if start_index != -1:
            end_index = start_index + 200  # Adjust the number of characters to extract as needed
            response_text = f"Based on the information I have about '{input}', here's what I found:\n\n"
            response_text += dataset_content[start_index:end_index]
            return response_text
        else:
            return "Sorry, I couldn't find relevant information in the dataset."
    else:
        return "Sorry, your input is out of context."

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="Chatbot", layout="wide")
st.sidebar.title("Chat History")

# Display chat history in reverse order (newest at the top)
for chat in reversed(st.session_state.chat_history):
    st.sidebar.write(chat)

st.header("Gemini Chatbot")
with st.form(key='input_form'):
    input = st.text_input("Input Prompt: ", key="input")
    submit = st.form_submit_button("Submit")

if submit:
    response = handle_input(input)
    st.write(response)
    # Debug: Print response
    print(f"Response: {response}")
    # Update chat history
    st.session_state.chat_history.append(f"Bot: {response}")
    st.session_state.chat_history.append(f"User: {input}")