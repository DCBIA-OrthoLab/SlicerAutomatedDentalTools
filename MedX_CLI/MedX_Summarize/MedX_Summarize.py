#!/usr/bin/env python-real
import argparse
import torch
import sys, os, time

from transformers import GenerationConfig

fpath = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(fpath)

from MedX_CLI_utils import create_chunks_from_paragraphs, extract_text_from_pdf, extract_text_from_word, clean_text, load_model_and_tokenizer


def generate_combined_summary(model, tokenizer, text, max_chunk_size=3500, model_max_tokens=1024):
    chunks = create_chunks_from_paragraphs(text, max_chunk_size=max_chunk_size)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    summaries = []
    
    generation_config = GenerationConfig(
        max_length=model_max_tokens,
        min_length=50,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=2.0,
        length_penalty=1.0,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
    )
    
    for chunk in chunks:
        prompt = f"Using the following note, extract structured key-value pairs about the patient's symptoms and diagnoses:\n\n{chunk}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=model_max_tokens).to(device)
        
        if inputs["input_ids"].shape[1] > model_max_tokens:
            print(f"WARNING: Chunk exceeded {model_max_tokens} tokens, truncating.")
            
        summary_ids = model.generate(
            inputs["input_ids"], 
            generation_config=generation_config
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        # print(f"Generated summary: {summary}")
        summaries.append(summary)
        

    final_summary = "\n----------------------------------------------------------------------------------------------------\n".join(summaries)
    return final_summary

def process_notes(notes_folder, output_folder, model, tokenizer, log_path):
    patient_files = {}
    
    for file_name in os.listdir(notes_folder):
        if not (file_name.endswith(".pdf") or file_name.endswith(".docx") or file_name.endswith(".txt")):
            continue
        patient_id = file_name.split("_")[0]
        if patient_id not in patient_files:
            patient_files[patient_id] = []
        patient_files[patient_id].append(file_name)

    for idx, (patient_id, files) in enumerate(patient_files.items()):
        print(f"Processing patient {patient_id}...")
        combined_text = ""
        for file_name in files:
            file_path = os.path.join(notes_folder, file_name)
            if file_name.endswith(".pdf"):
                combined_text += extract_text_from_pdf(file_path)
            elif file_name.endswith(".docx"):
                combined_text += extract_text_from_word(file_path)
            elif file_name.endswith(".txt"):
                with open(file_path, 'r', encoding='utf-8') as file:
                    combined_text += clean_text(file.read())

        summary = generate_combined_summary(model, tokenizer, combined_text)
        output_file_path = os.path.join(output_folder, f"{patient_id}_summary.txt")
        
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            print(f"Saved summary to {output_file_path}")
            output_file.write(f"{summary}\n")
            
        with open(log_path, "w+") as log_f:
                log_f.write(str(idx + 1))

def main(args):
    model, tokenizer = load_model_and_tokenizer(args.input_model)
    os.makedirs(args.output_folder, exist_ok=True)
    
    process_notes(args.input_notes, args.output_folder, model, tokenizer, args.log_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("input_notes", type=str)
    parser.add_argument("input_model", type=str)
    parser.add_argument("output_folder", type=str)
    parser.add_argument("log_path", type=str)

    args = parser.parse_args()

    main(args)
