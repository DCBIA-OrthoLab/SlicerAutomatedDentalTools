import re
import csv
import fitz
from docx import Document
from transformers import BartTokenizer, BartForConditionalGeneration

def load_model_and_tokenizer(model_path):
    """
    Load the BART model and tokenizer from the Hugging Face Transformers library.

    Args:
        model_path (str): Path to the pre-trained BART model.
        
    Returns:
        BartTokenizer, BartForConditionalGeneration: The tokenizer and model objects.
    """
    tokenizer = BartTokenizer.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path)
    return model, tokenizer

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using the PyMuPDF library (fitz).

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Cleaned text extracted from the PDF.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return clean_text(text)

def extract_text_from_word(docx_path):
    """
    Extracts text from a Word document using the python-docx library.

    Args:
        docx_path (str): Path to the Word document.

    Returns:
        str: Cleaned text extracted from the Word document.
    """
    doc = Document(docx_path)
    return "\n".join([clean_text(paragraph.text) for paragraph in doc.paragraphs])

def clean_text(text):
    """
    Cleans the text by replacing specific characters with their desired replacements.
    
    Args:
        text (str): The input text to clean.
    
    Returns:
        str: The cleaned text.
    """
    replacements = {
        "’": "'",
        "–": "-"
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def split_text_by_paragraphs(text):
    """
    Splits text into paragraphs based on likely paragraph boundaries.
    - Ensures list items stay together within the same paragraph.
    - Separates sections based on common section header keywords (e.g., "RADIOGRAPHIC EVALUATION").
    """
    # Normalize line breaks
    normalized_text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Define common section headers that should act as paragraph boundaries
    section_headers = ["CLINICAL EXAMINATION", "CLINICAL EVALUATION", "RADIOGRAPHIC EXAMINATION", "RADIOGRAPHIC EVALUATION"]
    header_pattern = r'(' + '|'.join(section_headers) + r')\.?'

    # Split based on double newlines, numbered lists, or section headers
    paragraphs = re.split(r'\n\s*\n|\n(?=\d+\.\s)|\n(?=\-)|\n(?=\*)|' + header_pattern, normalized_text)

    merged_paragraphs = []
    current_paragraph = ""

    for para in paragraphs:
        if para is None:
            continue

        para = para.strip()  # Remove leading and trailing whitespace

        if para in section_headers:
            # Treat section header as a separate paragraph
            if current_paragraph:
                merged_paragraphs.append(current_paragraph.strip())
            current_paragraph = para  # Start new paragraph with the section header
        elif re.match(r'^\d+\.\s|^[\-*]\s', para) or (current_paragraph and len(current_paragraph) < 150):
            # Add list items to the current paragraph
            current_paragraph += "\n" + para
        else:
            # Append current paragraph if it's not empty and reset for the new paragraph
            if current_paragraph:
                merged_paragraphs.append(current_paragraph.strip())
            current_paragraph = para  # Start a new paragraph

    # Add any remaining text as the last paragraph
    if current_paragraph:
        merged_paragraphs.append(current_paragraph.strip())

    return merged_paragraphs

def create_chunks_from_paragraphs(text, max_chunk_size=1800):
    section_headers = ["CLINICAL EXAMINATION", "CLINICAL EVALUATION", "RADIOGRAPHIC EXAMINATION", "RADIOGRAPHIC EVALUATION"]

    def split_to_sentences(paragraph, max_size):
        """
        Splits a paragraph into sentences that fit within the max size.
        """
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)  # Split on sentence boundaries
        chunk = ""
        chunks = []

        for sentence in sentences:
            if len(chunk) + len(sentence) + 1 <= max_size:
                chunk += sentence + " "
            else:
                if chunk:
                    chunks.append(chunk.strip())
                chunk = sentence + " "

        if chunk:
            chunks.append(chunk.strip())

        return chunks

    paragraphs = split_text_by_paragraphs(text)
    chunks = []
    current_chunk = ""

    i = 0
    while i < len(paragraphs):
        para = re.sub(r'\s{2,}', ' ', paragraphs[i])
        # Check if this paragraph starts with a section header
        if any(para.lower().startswith(header.lower()) for header in section_headers):
            # Accumulate the entire section: header + following non-header paragraphs.
            section_paragraphs = [para]
            j = i + 1
            while j < len(paragraphs):
                next_para = re.sub(r'\s{2,}', ' ', paragraphs[j])
                if any(next_para.lower().startswith(header.lower()) for header in section_headers):
                    break
                section_paragraphs.append(next_para)
                j += 1
            section_text = "\n\n".join(section_paragraphs)
            
            # If current chunk plus the whole section fits, add it.
            if len(current_chunk) + len(section_text) + 1 <= max_chunk_size:
                current_chunk += section_text + "\n\n"
            else:
                # Flush current_chunk if not empty.
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                # If the section itself is small enough, start a new chunk with it.
                if len(section_text) <= max_chunk_size:
                    current_chunk = section_text + "\n\n"
                else:
                    # Otherwise, process the section piece by piece.
                    for sec_para in section_paragraphs:
                        sec_para = re.sub(r'\s{2,}', ' ', sec_para)
                        if len(sec_para) <= max_chunk_size:
                            if len(current_chunk) + len(sec_para) + 1 <= max_chunk_size:
                                current_chunk += sec_para + "\n\n"
                            else:
                                if current_chunk:
                                    chunks.append(current_chunk.strip())
                                current_chunk = sec_para + "\n\n"
                        else:
                            # Split long paragraphs by sentences.
                            para_sentences = split_to_sentences(sec_para, max_chunk_size)
                            for sentence_chunk in para_sentences:
                                if len(current_chunk) + len(sentence_chunk) + 1 <= max_chunk_size:
                                    current_chunk += sentence_chunk + " "
                                else:
                                    if current_chunk:
                                        chunks.append(current_chunk.strip())
                                    current_chunk = sentence_chunk + " "
            i = j  # Move past the entire section.
        else:
            # Regular paragraph (non-header)
            if len(current_chunk) + len(para) + 1 <= max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                # If the paragraph itself is too long, split it.
                para_sentences = split_to_sentences(para, max_chunk_size)
                for sentence_chunk in para_sentences:
                    if len(current_chunk) + len(sentence_chunk) + 1 <= max_chunk_size:
                        current_chunk += sentence_chunk + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence_chunk + " "
            i += 1


    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def extract_key_value_pairs(text):
    """
    Extracts key-value pairs from text where the format is 'Key: Value'.

    Args:
        text (str): Text to extract key-value pairs from.

    Returns:
        dict: A dictionary containing extracted key-value pairs.
    """
    key_value_dict = {}
    pattern = r"([A-Za-z ]+):\s*(\d+)"
    matches = re.findall(pattern, text)
    
    for key, value in matches:
        key = key.strip()
        # TODO: Change the key check to the ones who needs integer 
        int_keys = ["Age", "Weight", "Height"]
        if key in int_keys:
            value = int(value)
        key_value_dict[key] = value
    
    return key_value_dict

def save_dict_to_csv(data_dict, output_file_path):
    """
    Save a dictionary to a CSV file.

    Args:
        data_dict (dict): The dictionary to save, where keys are column headers and values are their corresponding values.
        output_file_path (str): Path to the CSV file to save the data.
    """
    with open(output_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        
        # Write headers
        writer.writerow(["Key", "Value"])
        
        # Write key-value pairs
        for key, value in data_dict.items():
            writer.writerow([key, value])
    print(f"CSV file saved to {output_file_path}")