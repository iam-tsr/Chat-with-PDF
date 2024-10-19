import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import os
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

def get_gemini_response(input, image):
    if input != "":
        response = model.generate_content([input, image])
    else:
        response = model.generate_content(image)
    return response.text

# Initialize our Streamlit page
st.set_page_config(page_title="Chatbot", layout="wide")
st.sidebar.title("Chat History")
# st.sidebar.markdown("Use the sidebar to navigate through different functionalities.")

# Create a form for input and button
st.header("Gemini Chatbot")
with st.form(key='input_form'):
    input = st.text_input("Input Prompt: ", key="input")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "wpeg"])
    submit = st.form_submit_button("Tell me about the image")

image = ""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", width=300)

    
    st.markdown(
        """
        <style>
        button[title="View fullscreen"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# If submit button is clicked
if submit:
    response = get_gemini_response(input, image)
    # Display the response in a block-like box
    st.markdown(
        f"""<div style="border: 0.5px solid #3a3a3a; padding: 10px; border-radius: 5px;">{response}<div>""",

        unsafe_allow_html=True
    )

# Display chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if submit and input:
    st.session_state['chat_history'].append(("You", input))
    st.session_state['chat_history'].append(("Bot", response))

# st.sidebar.subheader("Chat History")
for role, text in st.session_state['chat_history']:
    st.sidebar.write(f"{role}: {text}")