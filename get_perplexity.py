import os
import json
import math 
import random
import sys
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm 

from prompts import *

DEVICE = "cuda"

@torch.no_grad() # Disable gradient calculation for inference
def get_logits_and_perplexity(model, tokenizer, text, device=DEVICE):
    """Calculates logits, token log-probabilities, and perplexity for a given text."""
    # Move model to device (if not done by device_map, though 8-bit usually handles it)
    # The models should already be on the target device via device_map="auto" and CUDA_VISIBLE_DEVICES.

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    # Note: Qwen2.5-3B is small enough that we can process the whole input at once.
    # For *much* larger inputs or models, batching would be necessary.
    
    # Forward pass
    outputs = model(input_ids)
    logits = outputs.logits
    
    # Shift logits and labels for standard language modeling perplexity calculation
    # Logits are [1, seq_len, vocab_size]
    shift_logits = logits[..., :-1, :].contiguous() # [1, seq_len-1, vocab_size]
    shift_labels = input_ids[..., 1:].contiguous()  # [1, seq_len-1]

    # Calculate log probabilities and NLL
    shift_logits = shift_logits.to(torch.float32) # Convert to full precision for log_softmax stability
    log_probs = torch.log_softmax(shift_logits, dim=-1) # [1, seq_len-1, vocab_size]
    
    # Gather the log probabilities of the actual target tokens
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1) # [1, seq_len-1]
    
    # Calculate Negative Log Likelihood (NLL)
    nll = -token_log_probs.mean()
    
    # Calculate Perplexity
    perplexity = torch.exp(nll).item()

    # The returned logits must be the *shifted* logits for subsequent combination
    # and must be moved to CPU to prevent OOM when accumulating.
    return shift_logits.to("cpu"), shift_labels.to("cpu"), perplexity




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Calculate perplexity for all combinations of base, expert, and anti-expert models.")
    parser.add_argument("--base_model_name_or_path", type=str, help="Path to the large-ins model.")
    parser.add_argument("--expert_model_name_or_path", type=str, help="Path to the small-cpt model.")
    parser.add_argument("--antiexpert_model_name_or_path", type=str, help="Path to the small-base model.")
    parser.add_argument("--load_in_8bit", action="store_true", help="Whether to load models in 8-bit precision for memory efficiency.")
    parser.add_argument("--max_test_example_num", type=int, default=50)
    parser.add_argument("--prompt_lang", type=str, default='en', help="Language for the prompts (e.g., 'en' for English).")

    parser.add_argument("--lang", type=str, required=True, help="Language code for the task (e.g., 'kk', 'zh', 'en').")
    parser.add_argument("--task_name", type=str, required=True, help="Name of the task (e.g., 'reading_comprehension', 'response_selection', 'text_classification', 'math', 'title_generation_200', 'translation_{lang}2en', 'translation_en2{lang}').")
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--exemplar_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)

    args = parser.parse_args()


    # --- Model Loading ---

    # Use `torch.bfloat16` for potentially better performance/stability than default
    # mixed-precision, while still using `load_in_8bit` for massive memory savings.

    # 1. Load the tokenizer once.
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path, trust_remote_code=True)


    print("Loading models...")

    common_load_args = {
        "load_in_8bit": args.load_in_8bit,
        "trust_remote_code": True,
        "device_map": "auto",
    }


    # Load Base Model (large-ins)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        **common_load_args
    )
    base_model.eval()

    # Load Expert Model (small-cpt)
    expert_model = AutoModelForCausalLM.from_pretrained(
        args.expert_model_name_or_path,
        **common_load_args
    )
    expert_model.eval()

    # Load Anti-Expert Model (small-base)
    antiexpert_model = AutoModelForCausalLM.from_pretrained(
        args.antiexpert_model_name_or_path,
        **common_load_args
    )
    antiexpert_model.eval()

    # Ensure all models have the same tokenizer vocabulary size (important for perplexity calculation)
    if base_model.get_input_embeddings().weight.size(0) != expert_model.get_input_embeddings().weight.size(0):
        expert_model.resize_token_embeddings(base_model.get_input_embeddings().weight.size(0))
        antiexpert_model.resize_token_embeddings(base_model.get_input_embeddings().weight.size(0))

    print("Models loaded successfully.")


    WEIGHT_COMBINATIONS = []
    for base_weight in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        for expert_weight in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            antiexpert_weight = 1 - base_weight - expert_weight
            # Only consider valid combinations (where antiexpert_weight is non-negative)
            # if antiexpert_weight >= 0: 
            WEIGHT_COMBINATIONS.append((base_weight, expert_weight, antiexpert_weight))



        
    print(f"\nProcessing task: {args.task_name}, language: {args.lang}")
    
    input_dataset = json.load(open(args.input_file, 'r', encoding='utf-8'))
    exemplar_dataset = json.load(open(args.exemplar_file, 'r', encoding='utf-8'))


    if args.task_name == 'reading_comprehension':
        num_exemplars = 3
        converted_dataset = convert_dataset_into_prompt_comprehension(input_dataset, exemplar_dataset, args.lang, num_exemplars, prompt_lang=args.prompt_lang)
    elif args.task_name == 'response_selection':
        num_exemplars = 3
        converted_dataset = convert_dataset_into_prompt_response(input_dataset, exemplar_dataset, args.lang, num_exemplars, prompt_lang=args.prompt_lang)
    elif args.task_name == 'text_classification':
        num_exemplars_per_label = 1
        max_passage_len = 384
        converted_dataset = convert_dataset_into_prompt_classification(input_dataset, exemplar_dataset, args.lang, num_exemplars_per_label, prompt_lang=args.prompt_lang, max_passage_len=max_passage_len)
    elif args.task_name == 'math':
        num_exemplars = 3
        converted_dataset = convert_dataset_into_prompt_math(input_dataset, exemplar_dataset, args.lang, num_exemplars, prompt_lang=args.prompt_lang)
    elif args.task_name == 'title_generation_200':
        num_exemplars = 3
        max_passage_len = 256
        converted_dataset = convert_dataset_into_prompt_title(input_dataset, exemplar_dataset, args.lang, num_exemplars, prompt_lang=args.prompt_lang, max_passage_len=max_passage_len)
    elif 'translation' in args.task_name:
        src_lang = args.task_name.split('_')[1].split('2')[0]
        tgt_lang = args.task_name.split('_')[1].split('2')[1]
        num_exemplar = 3
        converted_dataset = convert_dataset_into_prompt_translation(input_dataset, exemplar_dataset, src_lang, tgt_lang, num_exemplar, prompt_lang=args.prompt_lang)
        
    # sample a subset of the dataset for testing
    converted_dataset = random.sample(converted_dataset, args.max_test_example_num)
    

    # Initialize storage for average perplexities (for individual models) and NLL sums (for combinations)
    # We store the *sum* of NLLs/log_probs and the *count* to compute the overall average at the end.
    avg_base_perplexity_sum = 0
    avg_expert_perplexity_sum = 0
    avg_antiexpert_perplexity_sum = 0
    total_samples = 0

    # Store total NLL (Negative Log Likelihood) for each combination
    # We use a dictionary mapping (w_b, w_e, w_ae) to a list of NLLs
    nll_sums_by_combination = {combo: 0.0 for combo in WEIGHT_COMBINATIONS}


    print("\nProcessing dataset and calculating perplexities...")
    # Use tqdm for progress tracking
    for item in tqdm(converted_dataset, desc="Processing samples"):
        text = item['input']
        
        # --- Step 1: Get Logits, NLL, and Perplexity for each model ---
        
        # We use a revised helper function that returns (shifted_logits, shifted_labels, perplexity)
        # The returned logits must be moved to CPU immediately for temporary storage.
        base_logits, shift_labels, base_perplexity = get_logits_and_perplexity(base_model, tokenizer, text, device=DEVICE)
        expert_logits, _, expert_perplexity = get_logits_and_perplexity(expert_model, tokenizer, text, device=DEVICE)
        antiexpert_logits, _, antiexpert_perplexity = get_logits_and_perplexity(antiexpert_model, tokenizer, text, device=DEVICE)

        # Accumulate individual perplexity sums
        avg_base_perplexity_sum += base_perplexity
        avg_expert_perplexity_sum += expert_perplexity
        avg_antiexpert_perplexity_sum += antiexpert_perplexity
        total_samples += 1
        
        # --- Step 2: Calculate Combined Perplexity for all weights *immediately* ---
        # Logits are already on CPU from the helper function.

        # Ensure logits are float for numerical stability in combination
        base_logits = base_logits.to(torch.float32)
        expert_logits = expert_logits.to(torch.float32)
        antiexpert_logits = antiexpert_logits.to(torch.float32)
        shift_labels = shift_labels.to(DEVICE) # Move labels back to GPU for fast calculation
        
        for base_weight, expert_weight, antiexpert_weight in WEIGHT_COMBINATIONS:
            
            # Calculate combined logits on CPU (or GPU if memory allows, but CPU is safer)
            combined_logits = (base_weight * base_logits + 
                            expert_weight * expert_logits + 
                            antiexpert_weight * antiexpert_logits).to(DEVICE) # Move back to GPU for log_softmax

            # Calculation steps (now on GPU)
            combined_log_probs = torch.log_softmax(combined_logits, dim=-1)
            
            # The shift_labels were defined from the original input_ids.
            # Gather the log probabilities of the actual target tokens.
            combined_token_log_probs = combined_log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
            
            # Calculate NLL
            combined_nll = -combined_token_log_probs.mean() 
            
            # Accumulate the NLL for this weight combination
            nll_sums_by_combination[(base_weight, expert_weight, antiexpert_weight)] += combined_nll.item()

        # Clear temporary GPU memory from this sample
        del base_logits, expert_logits, antiexpert_logits, combined_logits, shift_labels
        torch.cuda.empty_cache()


    # --- Final Aggregation and Output ---

    # 1. Calculate overall average individual perplexities
    avg_base_perplexity = avg_base_perplexity_sum / total_samples
    avg_expert_perplexity = avg_expert_perplexity_sum / total_samples
    avg_antiexpert_perplexity = avg_antiexpert_perplexity_sum / total_samples

    print("\n--- Individual Model Averages ---")
    print(f"Base: {avg_base_perplexity:.4f}, Expert: {avg_expert_perplexity:.4f}, Anti-Expert: {avg_antiexpert_perplexity:.4f}")

    # 2. Calculate average combined perplexities
    perplexity_output = []
    print("\n--- Combined Perplexity Results ---")
    for combo, total_nll in nll_sums_by_combination.items():
        base_weight, expert_weight, antiexpert_weight = combo
        
        # Average NLL is the total NLL sum divided by the number of samples
        avg_combined_nll = total_nll / total_samples
        # Perplexity = exp(Avg_NLL)
        avg_combined_perplexity = math.exp(avg_combined_nll)

        result = {
            "base_weight": base_weight,
            "expert_weight": expert_weight,
            "antiexpert_weight": antiexpert_weight,
            "avg_base_perplexity": avg_base_perplexity,
            "avg_expert_perplexity": avg_expert_perplexity,
            "avg_antiexpert_perplexity": avg_antiexpert_perplexity,
            "avg_combined_perplexity": avg_combined_perplexity
        }
        perplexity_output.append(result)
        print(result)

    # 3. Save the final output
    json.dump(perplexity_output, open(args.output_file, 'w', encoding='utf-8'), indent=4)


    results = json.load(open(args.output_file, 'r', encoding='utf-8'))
    best_result = min(results, key=lambda x: x['avg_combined_perplexity'])
    print(f"Best config for {args.task_name} ({args.lang}):\n{best_result}")

