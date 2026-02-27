#!/usr/bin/env python-real

import sys, argparse, os, traceback

print("CNE_CLI.py run")

def main(args):
    # Arguments extraction
    notesFolder_input = args.notesFolder_input
    modelType = args.modelType
    notesType = args.notesType
    notesFolder_output = args.notesFolder_output
    modelPath = args.modelPath
    
    print("<filter-start><filter-name>Clinical Notes Extraction</filter-name></filter-start>", flush=True)
    
    # ---------------------------------------------------------
    # STEP 1 : Verification
    # ---------------------------------------------------------
    print("<filter-progress>0.20</filter-progress>", flush=True)
    print("<filter-comment>Checking model path...</filter-comment>", flush=True)
    
    if not os.path.exists(modelPath):
        print(f"ERROR: Model file not found at: {modelPath}", file=sys.stderr)
        sys.exit(1)

    # ---------------------------------------------------------
    # STEP 2 : Loading the model in memory
    # ---------------------------------------------------------
    print("<filter-progress>0.60</filter-progress>", flush=True)
    print(f"<filter-comment>Testing {modelType} model load...</filter-comment>", flush=True)
    
    try:
        from llama_cpp import Llama
        
        print(f"Initializing Llama engine with {modelPath}...")
        
        # Loading the model into memory (without running inference)
        llm = Llama(
            model_path=modelPath,
            n_ctx=128,      # Small context just for load testing
            n_threads=4,    # CPU threads
            verbose=False   # Keep logs clean
        )
        print("SUCCESS: Model loaded into memory successfully! (No inference run)")

    except ImportError:
        print("ERROR: 'llama-cpp-python' library is not installed in Slicer.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print("ERROR OCCURRED WHILE LOADING THE MODEL:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # ---------------------------------------------------------
    # FINISH : Progress to 100%
    # ---------------------------------------------------------
    print("<filter-progress>1.00</filter-progress>", flush=True)
    print("<filter-comment>Test complete!</filter-comment>", flush=True)

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