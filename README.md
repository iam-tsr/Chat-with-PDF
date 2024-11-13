# Chat with PDFs

## Overview
The **Chat with PDFs** application is a tool that allows users to upload PDF documents and interact with them by asking questions. The app processes the content of uploaded PDFs, splits the text into chunks, and stores it in a vectorized format to enable efficient similarity-based searches. Questions are then answered based on the document content using Google's Generative AI.

#### Deployed application - [Talk-with-PDF](https://chats-with-pdf.streamlit.app/)

## Features
- **PDF Upload**: Allows users to upload one or multiple PDF files.
- **Text Extraction and Chunking**: Extracts and chunks the text content of the PDFs for efficient processing.
- **Vector Storage**: Uses FAISS (Facebook AI Similarity Search) to store vectorized text chunks, enabling fast retrieval.
- **Generative AI for Q&A**: Leverages Google’s Generative AI for natural language processing to answer questions based on PDF content.
- **Streamlit Interface**: Provides an easy-to-use web interface for file upload, question entry, and answer display.

## Dependencies
To run this application locally on your system, you need to download few Python libraries:

These can be installed with:
```bash
pip install -r requirements.txt
```

## Environment Setup
1. **Google Generative AI API Key**: This application uses [Google's Generative AI API](https://ai.google.dev/). Store your API key in a `.env` file in the root directory:
   ```plaintext
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Usage
1. **Start the Streamlit App**:
   Run the application with:
   ```bash
   streamlit run main.py
   ```
   
2. **Upload PDF Files**:
   - Use the sidebar to upload your PDF files.
   - Click the "Submit & Process" button to process the files, extract and chunk the text, and store it for searching.

3. **Ask Questions**:
   - Type a question in the main input field to query the uploaded PDFs.
   - The application will return the most relevant answer based on the document content.

## Notes
- **Data Storage**: The vector index is saved locally as `faiss_index`, which allows for quick loading and querying of processed document data.
- **Generative AI Settings**: The model uses `gemini-pro` with a temperature setting of 0.3 to ensure relevant and accurate answers.