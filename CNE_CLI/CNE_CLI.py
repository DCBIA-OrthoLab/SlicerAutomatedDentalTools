#!/usr/bin/env python-real

import sys, argparse, os, traceback, glob, json
from pathlib import Path

import sys
import logging

# ===== Logging Configuration =====
logger = logging.getLogger("CNE_CLI")
logger.setLevel(logging.INFO)
logger.propagate = False
if logger.handlers:
    logger.handlers.clear()
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.info("CNE_CLI.py run")

def main(args):
    # Arguments extraction
    notesFolder_input = args.notesFolder_input
    modelType = args.modelType
    notesType = args.notesType
    notesFolder_output = args.notesFolder_output
    modelPath = args.modelPath
    
    print("<filter-start><filter-name>Clinical Notes Extraction</filter-name></filter-start>", flush=True)
    
    # ---------------------------------------------------------
    # STEP 1 : Model Path Validation
    # ---------------------------------------------------------
    print("<filter-progress>0.05</filter-progress>", flush=True)
    print("<filter-comment>Validating model path...</filter-comment>", flush=True)
    
    try:
        from llama_cpp import Llama
    except ImportError:
        logger.error("ERROR: 'llama-cpp-python' library is not installed in Slicer.", file=sys.stderr)
        sys.exit(1)
    
    # Validate that modelPath is provided and exists
    if not modelPath or not os.path.exists(modelPath):
        error_msg = f"ERROR: Model file not found or not provided: {modelPath}"
        logger.error(error_msg, file=sys.stderr)
        sys.exit(1)
    
    logger.info(f"Using model: {modelPath}")
    
    # ---------------------------------------------------------
    # STEP 2 : Verification
    # ---------------------------------------------------------
    print("<filter-progress>0.10</filter-progress>", flush=True)
    print("<filter-comment>Scanning input folder...</filter-comment>", flush=True)

    files_to_process = glob.glob(os.path.join(notesFolder_input, "*.txt"))
    
    if not files_to_process:
        logger.warning(f"WARNING: No .txt files found in {notesFolder_input}", file=sys.stderr)
        print("<filter-progress>1.00</filter-progress>", flush=True)
        sys.exit(0)

    # ---------------------------------------------------------
    # STEP 3 : Loading the model in memory
    # ---------------------------------------------------------
    print("<filter-progress>0.20</filter-progress>", flush=True)
    print(f"<filter-comment>Loading {modelType} model...</filter-comment>", flush=True)
    
    try:
        logger.info(f"Initializing Llama engine with {modelPath}...")

        if notesType == "TMJ":
            max_seq_length = 6144
        else:
            max_seq_length = 2048

        old_stderr = os.dup(sys.stderr.fileno())
        fd_devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(fd_devnull, sys.stderr.fileno())

        # Loading the model into memory with GPU support
        llm = Llama(
            model_path=modelPath,
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
                print(f"<filter-progress>{progress:.2f}</filter-progress>", flush=True)
                print(f"<filter-comment>Processing {filename} ({i+1}/{total_files})...</filter-comment>", flush=True)

                with open(file_path, 'r', encoding='utf-8') as f:
                    clinical_text = f.read()
                
                logger.info(f"Generating extraction for {filename}...")

                prompt = f"""<|start_header_id|>user<|end_header_id|>

            {clinical_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

            """
                output = llm(
                    prompt,
                    max_tokens=512,
                    temperature=0.3,
                    top_p=0.9,
                    echo=False
                )

                ai_response = output['choices'][0]['text'].strip()

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

                output_filename = f"Extraction_{filename}"
                output_filepath = os.path.join(notesFolder_output, output_filename)

                with open(output_filepath, 'w', encoding='utf-8') as f:
                    f.write(formatted_response)

                logger.info(f"Saved: {output_filepath}")
                successfully_processed += 1

            except Exception as e:
                logger.error(f"ERROR processing {filename}: {e}", file=sys.stderr)
                failed_files.append(filename)
                traceback.print_exc(file=sys.stderr)
                continue
        logger.info(f"Processing complete:")
        logger.info(f"Successfully processed: {successfully_processed}/{total_files}")
        if failed_files:
            logger.info(f"Failed files: {', '.join(failed_files)}")     

    except ImportError:
        logger.error("ERROR: 'llama-cpp-python' library is not installed in Slicer.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.error("ERROR OCCURRED DURING INFERENCE:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # ---------------------------------------------------------
    # FINISH : Progress to 100%
    # ---------------------------------------------------------
    print("<filter-progress>1.00</filter-progress>", flush=True)
    print("<filter-comment>All files processed successfully!</filter-comment>", flush=True)

    print("<filter-end><filter-name>Clinical Notes Extraction</filter-name></filter-end>", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('notesFolder_input', type=str)
    parser.add_argument('modelType', type=str)
    parser.add_argument("notesType", type=str)
    parser.add_argument('notesFolder_output', type=str)
    parser.add_argument('modelPath', type=str)

    args = parser.parse_args()
    main(args)
