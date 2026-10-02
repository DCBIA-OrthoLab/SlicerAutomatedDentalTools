#!/usr/bin/env python-real
import sys, argparse, os, traceback, glob, json


# ADTLib sits next to the modules in an installed build, in the directory Slicer
# already has on sys.path. A source tree has no such entry -- a module search
# path only gets there once Slicer finds a module in it, and ADT holds none --
# so the entry points walk up to the holder directory and add it themselves.
_adt_root = os.path.dirname(os.path.realpath(__file__))
while not os.path.isdir(os.path.join(_adt_root, "ADT", "ADTLib")) \
        and _adt_root != os.path.dirname(_adt_root):
    _adt_root = os.path.dirname(_adt_root)
if os.path.join(_adt_root, "ADT") not in sys.path:
    sys.path.append(os.path.join(_adt_root, "ADT"))

# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger
from ADTLib.progress_protocol import emit_fraction

logger = get_logger("CNE_CLI")

logger.info("CNE_CLI.py run")

INSTRUCTION_TMJ = "Using the following note, extract structured key-value pairs about the patient's symptoms and diagnoses:"

SUPPORTED_EXTENSIONS = (".txt", ".pdf", ".docx")

# Characters that vary between PDF/Word exports but never appeared in the
# plain-text notes used for training. Normalizing them keeps the text the
# model receives identical regardless of the source file format.
_TEXT_REPLACEMENTS = {
    "’": "'",   # right single quotation mark
    "‘": "'",   # left single quotation mark
    "“": '"',   # left double quotation mark
    "”": '"',   # right double quotation mark
    "–": "-",   # en dash
    "—": "-",   # em dash
    " ": " ",   # non-breaking space
}


def clean_text(text: str) -> str:
    """Normalize text so notes look the same to the model no matter their source format."""
    for old, new in _TEXT_REPLACEMENTS.items():
        text = text.replace(old, new)
    # Collapse excessive blank lines introduced by PDF/Word extraction
    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines = []
    for line in lines:
        if line or (cleaned_lines and cleaned_lines[-1]):
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract plain text from a PDF file using PyMuPDF (fitz)."""
    import fitz

    doc = fitz.open(pdf_path)
    try:
        text = "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()
    return clean_text(text)


def extract_text_from_docx(docx_path: str) -> str:
    """Extract plain text from a Word document using python-docx."""
    from docx import Document

    doc = Document(docx_path)
    return clean_text("\n".join(paragraph.text for paragraph in doc.paragraphs))


def extract_text(file_path: str) -> str:
    """Read a note regardless of its format (.txt/.pdf/.docx) and return normalized plain text."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    if ext == ".docx":
        return extract_text_from_docx(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        return clean_text(f.read())


def main(args):
    # Arguments extraction
    notes_folder_input = args.notesFolder_input
    notes_type = args.notesType
    notes_folder_output = args.notesFolder_output
    model_path = args.modelPath
    
    print("<filter-start><filter-name>Clinical Notes Extraction</filter-name></filter-start>", flush=True)
    
    # ---------------------------------------------------------
    # STEP 1 : Model Path Validation
    # ---------------------------------------------------------
    emit_fraction(0.05)
    print("<filter-comment>Validating model path...</filter-comment>", flush=True)
    
    try:
        from llama_cpp import Llama
    except ImportError:
        logger.error("ERROR: 'llama-cpp-python' library is not installed in Slicer.")
        sys.exit(1)

    try:
        import fitz  # noqa: F401  (PyMuPDF, required for .pdf notes)
    except ImportError:
        logger.error("ERROR: 'pymupdf' library is not installed in Slicer.")
        sys.exit(1)

    try:
        import docx  # noqa: F401  (python-docx, required for .docx notes)
    except ImportError:
        logger.error("ERROR: 'python-docx' library is not installed in Slicer.")
        sys.exit(1)

    # Validate that modelPath is provided and exists
    if not model_path or not os.path.exists(model_path):
        error_msg = f"ERROR: Model file not found or not provided: {model_path}"
        logger.error(error_msg)
        sys.exit(1)
    
    logger.info(f"Using model: {model_path}")
    
    # ---------------------------------------------------------
    # STEP 2 : Verification
    # ---------------------------------------------------------
    emit_fraction(0.10)
    print("<filter-comment>Scanning input folder...</filter-comment>", flush=True)

    files_to_process = []
    for ext in SUPPORTED_EXTENSIONS:
        files_to_process.extend(glob.glob(os.path.join(notes_folder_input, f"*{ext}")))

    if not files_to_process:
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        logger.warning(f"WARNING: No supported files ({supported}) found in {notes_folder_input}")
        emit_fraction(1.00)
        sys.exit(0)

    # ---------------------------------------------------------
    # STEP 3 : Loading the model in memory
    # ---------------------------------------------------------
    emit_fraction(0.20)
    print(f"<filter-comment>Loading model...</filter-comment>", flush=True)
    
    try:
        logger.info(f"Initializing Llama engine with {model_path}...")

        if notes_type == "TMJ":
            max_seq_length = 6144
        else:
            max_seq_length = 2048

        old_stderr = os.dup(sys.stderr.fileno())
        fd_devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(fd_devnull, sys.stderr.fileno())

        # Loading the model into memory with GPU support
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,    # Use GPU if available
            n_ctx=max_seq_length,
            verbose=False       # Keep logs clean
        )

        os.dup2(old_stderr, sys.stderr.fileno())
        os.close(old_stderr)
        os.close(fd_devnull)
        
        logger.info("SUCCESS: Model loaded into memory successfully!")

        # ---------------------------------------------------------
        # STEP 4 : Inference Loop (Processing Files)
        # ---------------------------------------------------------
        total_files = len(files_to_process)
        successfully_processed = 0
        failed_files = []
        
        for i, file_path in enumerate(files_to_process):
            filename = os.path.basename(file_path)
            
            try:
                progress = 0.20 + (0.75 * (i / total_files))
                emit_fraction(progress)
                print(f"<filter-comment>Processing {filename} ({i+1}/{total_files})...</filter-comment>", flush=True)

                clinical_text = extract_text(file_path)

                logger.info(f"Generating extraction for {filename}...")

                messages = []
                if notes_type.upper() == "TMJ":
                    messages.append({"role": "system", "content": INSTRUCTION_TMJ})
                messages.append({"role": "user", "content": clinical_text})

                output = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=500,
                    temperature=0.1,
                )

                ai_response = output['choices'][0]['message']['content'].strip()

                formatted_response = ""
                try:
                    start_idx = ai_response.find('{')
                    end_idx = ai_response.rfind('}') + 1
                    
                    if start_idx != -1 and end_idx != 0:
                        json_str = ai_response[start_idx:end_idx]
                        data = json.loads(json_str)
                        
                        if "extraction" in data:
                            data = data["extraction"]
                            
                        for key, value in data.items():
                            formatted_response += f"{key} : {value}\n"
                    else:
                        formatted_response = ai_response
                        
                except Exception as e:
                    logger.warning(f"Warning: Could not format JSON for {filename}: {e}")
                    formatted_response = ai_response

                output_filename = f"Extraction_{os.path.splitext(filename)[0]}.txt"
                output_filepath = os.path.join(notes_folder_output, output_filename)

                with open(output_filepath, 'w', encoding='utf-8') as f:
                    f.write(formatted_response)

                logger.info(f"Saved: {output_filepath}")
                successfully_processed += 1

            except Exception as e:
                logger.error(f"ERROR processing {filename}: {e}")
                failed_files.append(filename)
                traceback.print_exc(file=sys.stderr)
                continue
        logger.info(f"Processing complete:")
        logger.info(f"Successfully processed: {successfully_processed}/{total_files}")
        if failed_files:
            logger.info(f"Failed files: {', '.join(failed_files)}")

    except ImportError:
        logger.error("ERROR: 'llama-cpp-python' library is not installed in Slicer.")
        sys.exit(1)
    except Exception as e:
        # `e` was bound and never used, and the traceback went to sys.stderr
        # alone -- which Slicer does not put in the log the operator reads. All
        # that reached it was the header, so a CNE run that failed said
        # "ERROR OCCURRED DURING INFERENCE:" and nothing else, on every machine
        # and for every cause. Measured on 2026-09-29: the load of a 7B GGUF
        # failed identically on main and on adt/integration, and neither run
        # could say why.
        logger.error("ERROR OCCURRED DURING INFERENCE: %s: %s",
                     type(e).__name__, e)
        logger.error("%s", traceback.format_exc())
        sys.exit(1)

    # ---------------------------------------------------------
    # FINISH : Progress to 100%
    # ---------------------------------------------------------
    emit_fraction(1.00)
    print("<filter-comment>All files processed successfully!</filter-comment>", flush=True)

    print("<filter-end><filter-name>Clinical Notes Extraction</filter-name></filter-end>", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('notesFolder_input', type=str)
    parser.add_argument("notesType", type=str)
    parser.add_argument('notesFolder_output', type=str)
    parser.add_argument('modelPath', type=str)

    args = parser.parse_args()
    main(args)