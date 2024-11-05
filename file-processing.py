from PyPDF2 import PdfReader
from docx import Document
import os
import pypandoc

# Example usage
input_file = '/mnt/Linux/Projects/Custom-Chatbot/downloads/wbs for IC.docx'
output_txt = '/mnt/Linux/Projects/Custom-Chatbot/example.txt'

def main():
    process_file(input_file, output_txt)

def extract_text_from_docx(input_file, output_txt):
    try:
        # Load the .docx file
        doc = Document(input_file)
        
        # Open the output text file for writing
        with open(output_txt, 'w') as txt_file:
            # Write each paragraph to the text file
            for para in doc.paragraphs:
                txt_file.write(para.text + '\n')
                
        print(f"Text successfully extracted to {output_txt}")
    
    except Exception as e:
        print(f"Error processing DOCX file '{input_file}': {e}")

def extract_text_from_pdf(input_file, output_txt):
    try:
        with open(input_file, 'rb') as pdf_file:
            reader = PdfReader(pdf_file)
            with open(output_txt, 'w') as txt_file:
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        txt_file.write(text + '\n')
                    else:
                        print(f"Warning: Could not extract text from one or more pages.")
    except Exception as e:
        print(f"Error processing PDF file: {e}")

def process_file(input_path, output_path):
    _, ext = os.path.splitext(input_path)
    if ext.lower() == '.pdf':
        extract_text_from_pdf(input_path, output_path)
    elif ext.lower() in ['.doc', '.docx']:
        extract_text_from_docx(input_path, output_path)
    else:
        print(f"Unsupported file extension: {ext}")

# Run the example
if __name__ == "__main__":
    main()
