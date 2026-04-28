import json
import argparse
import os


from utils import (
    generate_completions,
    load_trimix_model_and_tokenizer,
)
from prompts import *


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, required=True)

    parser.add_argument("--base_model_name_or_path", type=str, required=True, help="Path to the large-ins model.")
    parser.add_argument("--expert_model_name_or_path", type=str, required=True, help="Path to the small-cpt model.")
    parser.add_argument("--antiexpert_model_name_or_path", type=str, required=True, help="Path to the small-base model.")
    parser.add_argument('--load_in_8bit', action='store_true')
    parser.add_argument('--base_weight', type=float, default=1.0)
    parser.add_argument('--expert_weight', type=float, default=1.0)
    parser.add_argument('--plausibility_model', type=str, default="none")
    parser.add_argument('--plausibility_alpha', type=float, default=0.0)
    

    parser.add_argument('--exemplar_file', type=str, default=None)
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)

    parser.add_argument('--max_new_tokens', type=int, default=250) 
    parser.add_argument('--batch_size', type=int, default=1) # recommend using 1 to avoid potential errors

    # for debugging
    parser.add_argument('--print_inference_result', action='store_true')
    parser.add_argument('--max_test_example_num', type=int, default=-1) 

    # args for task
    parser.add_argument('--eval_lang', type=str, default='bo')
    parser.add_argument('--num_exemplar', type=int, default=3)
    parser.add_argument('--prompt_lang', type=str, default='zh')
    parser.add_argument('--num_exemplar_per_label', type=int, default=3) # for text classification only
    parser.add_argument('--src_lang', type=str, default='zh') # for translation only
    parser.add_argument('--tgt_lang', type=str, default='mn') # for translation only
    parser.add_argument('--max_passage_len', type=int, default=1024) # for title generation and text classification only
    

    args = parser.parse_args()

    # get the folder of the output file
    output_folder = '/'.join(args.output_file.split('/')[:-1])
    if len(output_folder) == 0:
        output_folder = '.'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if os.path.exists(args.output_file):
        print(f"Warning: the output file {args.output_file} already exists. Stop running.")
        exit(0)
        # print(f"Warning: the output file {args.output_file} already exists, save the new results to {args.output_file}.new")
        # args.output_file += '.new'

    # pre-processing
    input_dataset = json.load(open(args.input_file, 'r', encoding='utf-8'))
    exemplar_dataset = json.load(open(args.exemplar_file, 'r', encoding='utf-8')) if args.exemplar_file is not None else None

    if args.prompt_lang not in ['en', 'zh']:
        raise NotImplementedError

    if args.task == 'math':
        converted_dataset = convert_dataset_into_prompt_math(input_dataset, exemplar_dataset, args.eval_lang, args.num_exemplar, prompt_lang=args.prompt_lang)
    elif args.task == 'reading_comprehension':
        converted_dataset = convert_dataset_into_prompt_comprehension(input_dataset, exemplar_dataset, args.eval_lang, args.num_exemplar, prompt_lang=args.prompt_lang)
    elif args.task == 'response_selection':
        converted_dataset = convert_dataset_into_prompt_response(input_dataset, exemplar_dataset, args.eval_lang, args.num_exemplar, prompt_lang=args.prompt_lang)
    elif args.task == 'text_classification':
        converted_dataset = convert_dataset_into_prompt_classification(input_dataset, exemplar_dataset, args.eval_lang, num_exemplar_per_label=args.num_exemplar_per_label, max_passage_len=args.max_passage_len, prompt_lang=args.prompt_lang)
    elif args.task == 'title_generation':
        converted_dataset = convert_dataset_into_prompt_title(input_dataset, exemplar_dataset, args.eval_lang, args.num_exemplar, args.max_passage_len, prompt_lang=args.prompt_lang)
    elif args.task == 'translation':
        converted_dataset = convert_dataset_into_prompt_translation(input_dataset, exemplar_dataset, args.src_lang, args.tgt_lang, args.num_exemplar, prompt_lang=args.prompt_lang)
    else:
        raise NotImplementedError
    
    # for debugging, only use a small part of the dataset
    if args.max_test_example_num > 0:
        converted_dataset = converted_dataset[:args.max_test_example_num]

    print("prompt sample:", converted_dataset[0]['input'])


    # load model 
    model, tokenizer = load_trimix_model_and_tokenizer(
            args.base_model_name_or_path,
            args.expert_model_name_or_path,
            args.antiexpert_model_name_or_path,
            chat_response_prefix="",
            base_weight=args.base_weight,
            expert_weight=args.expert_weight,
            load_in_8bit=args.load_in_8bit,
            use_fast_tokenizer=True,
            plausibility_model=args.plausibility_model,
            plausibility_alpha=args.plausibility_alpha,
        )
    print("model loaded")


    prompts = [item['input'] for item in converted_dataset]

    outputs = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        do_sample=False,
        return_logits_for_analysis=False,
        stop_id_sequences=[tokenizer.encode("\n\n", add_special_tokens=False), tokenizer.encode("\n", add_special_tokens=False), tokenizer.encode(".\n\n", add_special_tokens=False), tokenizer.encode("!\n\n", add_special_tokens=False), tokenizer.encode("?\n\n", add_special_tokens=False), tokenizer.encode("<|endoftext|>", add_special_tokens=False),]
    ) # for simplicity, we only consider the first line of the output as the final answer, and remove the special tokens in the output. You can modify this part to better fit your task.

    output_results = {}
    for i in range(len(converted_dataset)):
        qid = converted_dataset[i]['id']
        
        output = remove_special_tokens(outputs[i].strip().split('\n')[0])
        # print(f"qid: {qid}, output: {output}")
        output_results[qid] = output

    # save results
    json.dump(output_results, open(args.output_file, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
