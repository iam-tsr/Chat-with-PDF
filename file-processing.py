from text_extractor import text_extraction
from doc_parser import docx_parser
from docx import Document
import mimetypes
import os


filepath = "/mnt/Linux/Projects/Custom-Chatbot/downloads/Roadmap.doc"

mime_type = mimetypes.guess_type(filepath)

if mime_type == 'application/pdf':
    print("Loading file as PDF file format")
    output_text = text_extraction(filepath)

elif mime_type == 'application/doc' or mime_type == 'application/docx':
    print("Loading file as .doc or .docx file format")
    output_text = Document(filepath)

else:
    print("Unsupported file format")
    output_text = ""

# Write the output text to a file
with open('/mnt/Linux/Projects/Custom-Chatbot/example.txt', 'w', encoding='utf-8') as file:
    file.write(output_text)