#!/usr/bin/env python-real

import sys, argparse, os, traceback, glob, json, urllib.request
from pathlib import Path

print("CNE_CLI.py run")

# ============================================================================
# HUGGING FACE MODEL CONFIGURATION
# ============================================================================
HF_REPO_ID = "dcbia/Meta-Llama-3.1-8B-Instruct-Ortho"
HF_MODEL_FILE = "model-q4_0.gguf"
HF_LOCAL_NAME = "Meta-Llama-3.1-8B-Ortho.gguf"
# ============================================================================

def main(args):
    # Arguments extraction
    notesFolder_input = args.notesFolder_input
    modelType = args.modelType
    notesType = args.notesType
    notesFolder_output = args.notesFolder_output
    modelPath = args.modelPath
    
    print("<filter-start><filter-name>Clinical Notes Extraction</filter-name></filter-start>", flush=True)
    
    # ---------------------------------------------------------
    # STEP 1 : Model Path Resolution & Download from HF if needed
    # ---------------------------------------------------------
    print("<filter-progress>0.05</filter-progress>", flush=True)
    print("<filter-comment>Resolving model path...</filter-comment>", flush=True)
    
    try:
        from llama_cpp import Llama
    except ImportError:
        print("ERROR: 'llama-cpp-python' library is not installed in Slicer.", file=sys.stderr)
        sys.exit(1)
    
    # Setup cache directory for model
    project_root = Path.home() / ".cache" / "medical-llm"
    project_root.mkdir(parents=True, exist_ok=True)
    
    # If modelPath is not provided or doesn't exist, download from Hugging Face
    if not modelPath or not os.path.exists(modelPath):
        modelPath = str(project_root / HF_LOCAL_NAME)
        
        if not Path(modelPath).exists():
            print("<filter-progress>0.08</filter-progress>", flush=True)
            print("<filter-comment>Downloading model from Hugging Face...</filter-comment>", flush=True)
            modelUrl = f"https://huggingface.co/{HF_REPO_ID}/resolve/main/{HF_MODEL_FILE}"
            
            try:
                print(f"Downloading from: {modelUrl}")
                
                def progress_hook(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    if total_size > 0:
                        percent = min(100, int(downloaded / total_size * 100))
                        mb_downloaded = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        print(f"Progress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='\r')
                
                urllib.request.urlretrieve(modelUrl, modelPath, progress_hook)
                print("\nModel downloaded successfully!")
            except Exception as e:
                print(f"\nERROR downloading model: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"Model already cached at: {modelPath}")
    
    # ---------------------------------------------------------
    # STEP 2 : Verification
    # ---------------------------------------------------------
    print("<filter-progress>0.10</filter-progress>", flush=True)
    print("<filter-comment>Scanning input folder...</filter-comment>", flush=True)
    
    if not os.path.exists(modelPath):
        print(f"ERROR: Model file not found at: {modelPath}", file=sys.stderr)
        sys.exit(1)

    # Récupérer tous les fichiers .txt du dossier d'entrée
    files_to_process = glob.glob(os.path.join(notesFolder_input, "*.txt"))
    
    if not files_to_process:
        print(f"WARNING: No .txt files found in {notesFolder_input}", file=sys.stderr)
        print("<filter-progress>1.00</filter-progress>", flush=True)
        sys.exit(0)

    # ---------------------------------------------------------
    # STEP 3 : Loading the model in memory
    # ---------------------------------------------------------
    print("<filter-progress>0.20</filter-progress>", flush=True)
    print(f"<filter-comment>Loading {modelType} model...</filter-comment>", flush=True)
    
    try:
        print(f"Initializing Llama engine with {modelPath}...")

        if notesType == "TMJ":
            max_seq_length = 6144
        else:
            max_seq_length = 2048
        
        # --- ASTUCE POUR RENDRE LE C++ TOTALEMENT SILENCIEUX ---
        old_stderr = os.dup(sys.stderr.fileno())
        fd_devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(fd_devnull, sys.stderr.fileno())
        # -------------------------------------------------------

        # Loading the model into memory with GPU support
        llm = Llama(
            model_path=modelPath,
            n_gpu_layers=-1,    # Use GPU if available
            n_ctx=max_seq_length,      
            verbose=False       # Keep logs clean
        )
        
        # --- RESTAURATION DU CANAL D'ERREUR ---
        os.dup2(old_stderr, sys.stderr.fileno())
        os.close(old_stderr)
        os.close(fd_devnull)
        # -------------------------------------------------------
        
        print("SUCCESS: Model loaded into memory successfully!")

        # ---------------------------------------------------------
        # STEP 4 : Inference Loop (Processing Files)
        # ---------------------------------------------------------
        total_files = len(files_to_process)
        successfully_processed = 0
        failed_files = []
        
        for i, file_path in enumerate(files_to_process):
            filename = os.path.basename(file_path)
            
            try:
                # Mise à jour fluide de la barre de progression Slicer
                progress = 0.20 + (0.75 * (i / total_files))
                print(f"<filter-progress>{progress:.2f}</filter-progress>", flush=True)
                print(f"<filter-comment>Processing {filename} ({i+1}/{total_files})...</filter-comment>", flush=True)
                
                # 1. Lecture du document brut
                with open(file_path, 'r', encoding='utf-8') as f:
                    clinical_text = f.read()
                
                print(f"Generating extraction for {filename}...")
                
                # 2. Format prompt using Llama 3.1 chat template
                prompt = f"""<|start_header_id|>user<|end_header_id|>

            {clinical_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

            """
                
                # 3. Lancement de l'inférence avec le template chat
                output = llm(
                    prompt,
                    max_tokens=512,   
                    temperature=0.3,
                    top_p=0.9,
                    echo=False        
                )
                
                # 3. Récupération du résultat brut
                ai_response = output['choices'][0]['text'].strip()
                
                # --- FORMATTAGE DU JSON EN LISTE LISIBLE ---
                formatted_response = ""
                try:
                    # On cherche où commence et où finit le JSON 
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
                    print(f"Warning: Could not format JSON for {filename}: {e}")
                    formatted_response = ai_response
                # -----------------------------------------------------
                
                # 4. Sauvegarde dans le dossier de sortie
                output_filename = f"Extraction_{filename}"
                output_filepath = os.path.join(notesFolder_output, output_filename)
                
                with open(output_filepath, 'w', encoding='utf-8') as f:
                    f.write(formatted_response)
                    
                print(f"✓ Saved: {output_filepath}")
                successfully_processed += 1
                
            except Exception as e:
                print(f"✗ ERROR processing {filename}: {e}", file=sys.stderr)
                failed_files.append(filename)
                traceback.print_exc(file=sys.stderr)
                # Continue avec le fichier suivant au lieu d'arrêter complètement
                continue
        
        # Résumé final
        print(f"\n{'='*50}")
        print(f"Processing complete:")
        print(f"  ✓ Successfully processed: {successfully_processed}/{total_files}")
        if failed_files:
            print(f"  ✗ Failed files: {', '.join(failed_files)}")
        print(f"{'='*50}\n")      

    except ImportError:
        print("ERROR: 'llama-cpp-python' library is not installed in Slicer.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print("ERROR OCCURRED DURING INFERENCE:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # ---------------------------------------------------------
    # FINISH : Progress to 100%
    # ---------------------------------------------------------
    print("<filter-progress>1.00</filter-progress>", flush=True)
    print("<filter-comment>All files processed successfully!</filter-comment>", flush=True)

    print("<filter-end><filter-name>Clinical Notes Extraction</filter-name></filter-end>", flush=True)
    print("\nCNE_CLI.py done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('notesFolder_input', type=str)
    parser.add_argument('modelType', type=str)
    parser.add_argument("notesType", type=str)
    parser.add_argument('notesFolder_output', type=str)
    parser.add_argument('modelPath', type=str)

    args = parser.parse_args()
    main(args)
