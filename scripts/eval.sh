seed=1 
model_name=$1
eval_lang=$2
prompt_lang="en"

# response selection
echo "==== Response Selection ===="
python eval.py \
    --task response_selection \
    --input_file data/response_selection/${eval_lang}/test.json \
    --pred_file inference_results/${model_name}/response_selection/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test.json \
    --metrics_output_file inference_results/${model_name}/response_selection/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test_metrics.json 


# reading comprehension
echo "==== Reading Comprehension ===="
python eval.py \
    --task reading_comprehension \
    --input_file data/reading_comprehension/${eval_lang}/test.json \
    --pred_file inference_results/${model_name}/reading_comprehension/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test.json \
    --metrics_output_file inference_results/${model_name}/reading_comprehension/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test_metrics.json



# text classification
echo "==== Text Classification ===="
python eval.py \
    --task text_classification \
    --input_file data/text_classification/${eval_lang}/test.json \
    --pred_file inference_results/${model_name}/text_classification/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test.json \
    --metrics_output_file inference_results/${model_name}/text_classification/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test_metrics.json \
    --label_lang ${prompt_lang}


# title generation
echo "==== Title Generation ===="
python eval.py \
    --task title_generation \
    --input_file data/title_generation_200/${eval_lang}/test.json \
    --pred_file inference_results/${model_name}/title_generation_200/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test.json \
    --metrics_output_file inference_results/${model_name}/title_generation_200/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test_metrics.json


# translation (dialogue)
echo "==== Translation (Dialogue) ===="
# eval_lang -> prompt_lang
echo "Translation: ${eval_lang} -> ${prompt_lang}"
python eval.py \
    --task translation \
    --tgt_lang ${prompt_lang} \
    --input_file data/translation_dialogue/${eval_lang}/test.json \
    --pred_file inference_results/${model_name}/translation_dialogue/${eval_lang}/${prompt_lang}-prompt_seed${seed}_${eval_lang}2${prompt_lang}_test.json \
    --metrics_output_file inference_results/${model_name}/translation_dialogue/${eval_lang}/${prompt_lang}-prompt_seed${seed}_${eval_lang}2${prompt_lang}_test_metrics.json

# prompt_lang -> eval_lang
echo "Translation: ${prompt_lang} -> ${eval_lang}"
python eval.py \
    --task translation \
    --tgt_lang ${eval_lang} \
    --input_file data/translation_dialogue/${eval_lang}/test.json \
    --pred_file inference_results/${model_name}/translation_dialogue/${eval_lang}/${prompt_lang}-prompt_seed${seed}_${prompt_lang}2${eval_lang}_test.json \
    --metrics_output_file inference_results/${model_name}/translation_dialogue/${eval_lang}/${prompt_lang}-prompt_seed${seed}_${prompt_lang}2${eval_lang}_test_metrics.json

# math
echo "==== Math ===="
python eval.py \
    --task math \
    --input_file data/math/${eval_lang}/test.json \
    --pred_file inference_results/${model_name}/math/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test.json \
    --metrics_output_file inference_results/${model_name}/math/${eval_lang}/${prompt_lang}-prompt_seed${seed}_test_metrics.json