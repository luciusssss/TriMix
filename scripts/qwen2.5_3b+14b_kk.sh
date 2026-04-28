base_model_name_or_path="Qwen/Qwen2.5-14B-Instruct"
expert_model_name_or_path="pkupie/Qwen2.5-3B-kk-cpt"
antiexpert_model_name_or_path="Qwen/Qwen2.5-3B"
model_name="qwen2.5_trimix_3b+14b_kk"
prompt_lang="en"
eval_lang="kk"

seed=1

python infer.py \
    --task response_selection \
    --base_model_name_or_path ${base_model_name_or_path} \
    --expert_model_name_or_path ${expert_model_name_or_path} \
    --antiexpert_model_name_or_path ${antiexpert_model_name_or_path} \
    --exemplar_file data/response_selection/${eval_lang}/train_${seed}.json \
    --input_file data/response_selection/${eval_lang}/test.json \
    --output_file inference_results/${model_name}/response_selection/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test.json \
    --max_new_tokens 2 \
    --eval_lang ${eval_lang} \
    --num_exemplar 5 \
    --prompt_lang ${prompt_lang} \
    --load_in_8bit \
    --base_weight 0.1 \
    --expert_weight 1.0


python infer.py \
    --task reading_comprehension \
    --base_model_name_or_path ${base_model_name_or_path} \
    --expert_model_name_or_path ${expert_model_name_or_path} \
    --antiexpert_model_name_or_path ${antiexpert_model_name_or_path} \
    --exemplar_file data/reading_comprehension/${eval_lang}/train_${seed}.json \
    --input_file data/reading_comprehension/${eval_lang}/test.json \
    --output_file inference_results/${model_name}/reading_comprehension/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test.json \
    --max_new_tokens 2 \
    --eval_lang ${eval_lang} \
    --num_exemplar 5 \
    --prompt_lang ${prompt_lang} \
    --load_in_8bit \
    --base_weight 0.1 \
    --expert_weight 1.0 

python infer.py \
    --task text_classification \
    --base_model_name_or_path ${base_model_name_or_path} \
    --expert_model_name_or_path ${expert_model_name_or_path} \
    --antiexpert_model_name_or_path ${antiexpert_model_name_or_path} \
    --exemplar_file data/text_classification/${eval_lang}/train_${seed}.json \
    --input_file data/text_classification/${eval_lang}/test.json \
    --output_file inference_results/${model_name}/text_classification/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test.json \
    --max_new_tokens 2 \
    --eval_lang ${eval_lang} \
    --num_exemplar 5 \
    --prompt_lang ${prompt_lang} \
    --load_in_8bit \
    --base_weight 0.1 \
    --expert_weight 0.9

python infer.py \
    --task title_generation \
    --base_model_name_or_path ${base_model_name_or_path} \
    --expert_model_name_or_path ${expert_model_name_or_path} \
    --antiexpert_model_name_or_path ${antiexpert_model_name_or_path} \
    --exemplar_file data/title_generation_200/${eval_lang}/train_${seed}.json \
    --input_file data/title_generation_200/${eval_lang}/test.json \
    --output_file inference_results/${model_name}/title_generation_200/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test.json \
    --max_new_tokens 250 \
    --eval_lang ${eval_lang} \
    --num_exemplar 3 \
    --prompt_lang ${prompt_lang} \
    --load_in_8bit \
    --base_weight 0.1 \
    --expert_weight 1.0

python infer.py \
    --task math \
    --base_model_name_or_path ${base_model_name_or_path} \
    --expert_model_name_or_path ${expert_model_name_or_path} \
    --antiexpert_model_name_or_path ${antiexpert_model_name_or_path} \
    --exemplar_file data/math/${eval_lang}/train_${seed}.json \
    --input_file data/math/${eval_lang}/test.json \
    --output_file inference_results/${model_name}/math/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test.json \
    --max_new_tokens 250 \
    --eval_lang ${eval_lang} \
    --num_exemplar 5 \
    --prompt_lang ${prompt_lang} \
    --load_in_8bit \
    --base_weight 0.1 \
    --expert_weight 0.9

python infer.py\
    --task translation \
    --base_model_name_or_path ${base_model_name_or_path} \
    --expert_model_name_or_path ${expert_model_name_or_path} \
    --antiexpert_model_name_or_path ${antiexpert_model_name_or_path} \
    --exemplar_file data/translation_dialogue/${eval_lang}/train_${seed}.json \
    --input_file data/translation_dialogue/${eval_lang}/test.json \
    --output_file inference_results/${model_name}/translation_dialogue/${eval_lang}/${prompt_lang}-prompt_seed${seed}_kk2en_test.json \
    --max_new_tokens 200 \
    --num_exemplar 5 \
    --src_lang ${eval_lang} \
    --tgt_lang ${prompt_lang} \
    --eval_lang ${eval_lang} \
    --prompt_lang ${prompt_lang} \
    --load_in_8bit \
    --base_weight 0.1 \
    --expert_weight 1.0 


python infer.py \
    --task translation \
    --base_model_name_or_path ${base_model_name_or_path} \
    --expert_model_name_or_path ${expert_model_name_or_path} \
    --antiexpert_model_name_or_path ${antiexpert_model_name_or_path} \
    --exemplar_file data/translation_dialogue/${eval_lang}/train_${seed}.json \
    --input_file data/translation_dialogue/${eval_lang}/test.json \
    --output_file inference_results/${model_name}/translation_dialogue/${eval_lang}/${prompt_lang}-prompt_seed${seed}_en2kk_test.json \
    --max_new_tokens 200 \
    --num_exemplar 5 \
    --src_lang ${prompt_lang} \
    --tgt_lang ${eval_lang} \
    --eval_lang ${eval_lang} \
    --prompt_lang ${prompt_lang} \
    --load_in_8bit \
    --base_weight 0.3 \
    --expert_weight 1.0

