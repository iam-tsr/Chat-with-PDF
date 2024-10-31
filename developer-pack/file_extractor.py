import os
import PyPDF2
import pandas as pd
from docx import Document

def load_dataset(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    content = ""
    if ext == ".pdf":
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                content += page.extract_text()
    elif ext == ".txt":
        with open(file_path, "r") as file:
            content = file.read()
    elif ext == ".docx":
        doc = Document(file_path)
        for para in doc.paragraphs:
            content += para.text
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path)
        content = df.to_string()
    return content