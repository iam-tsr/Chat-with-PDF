# Chat with PDF

## Overview
The **Chat with PDFs** application is a tool that allows users to upload PDF documents and interact with them by asking questions. The app processes the content of uploaded PDFs, splits the text into chunks, and stores it in a vectorized format to enable efficient similarity-based searches. Questions are then answered based on the document content using Google's Generative AI.

**Deployed application** - [Talk-with-PDF](https://chats-with-pdf.streamlit.app/)

**Google Generative AI API Key**: This application uses [Google's Generative AI API](https://ai.google.dev/).

## Features
- **PDF Upload**: Allows users to upload one or multiple PDF files.
- **Text Extraction and Chunking**: Extracts and chunks the text content of the PDFs for efficient processing.
- **Vector Storage**: Uses FAISS (Facebook AI Similarity Search) to store vectorized text chunks, enabling fast retrieval.
- **Generative AI for Q&A**: Leverages Google’s Generative AI for natural language processing to answer questions based on PDF content.
- **Streamlit Interface**: Provides an easy-to-use web interface for file upload, question entry, and answer display.

## Local Installation
To run this application locally on your system, you need to download few Python libraries:

**These can be installed with**:
```bash
pip install -r requirements.txt
```
**Start the Streamlit App**:
   Run the application with:
   ```bash
   streamlit run main.py
   ```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENCE) file for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any changes.

## Contact
For any inquiries, please contact tusharsoni.info@gmail.com.
