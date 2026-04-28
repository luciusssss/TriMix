# Efficient Low-Resource Language Adaptation via Multi-Source Dynamic Logit Fusion

This repository contains the code for the paper "Efficient Low-Resource Language Adaptation via Multi-Source Dynamic Logit Fusion" (ACL 2026) (https://arxiv.org/abs/2604.18106). 

## Quick Start

1. Install the required packages:
```bash
pip install -r requirements.txt
```
Install the package required by multilingual ROUGE scoring. See [https://github.com/csebuetnlp/xl-sum/tree/master/multilingual_rouge_scoring](https://github.com/csebuetnlp/xl-sum/tree/master/multilingual_rouge_scoring) for more details.
```bash

2. Unzip the evaluation datasets (MiLiC-Eval) with the password `milic`.
```
unzip -P milic data.zip  
```

The expected structure is as follows:
```data/
├── reading_comprehension/
│   ├── kk/
│   │   └── test.json
...
```

3. (Optional) If you want to get the best parameters for logit fusion, you can run the following command to get the perplexity-selected results:
```bash
python get_perplexity.py \
--base_model_name_or_path Qwen/Qwen2.5-7B-Instruct \
# path_to_large-ins_model 
--expert_model_name_or_path pkupie/Qwen2.5-1.5B-kk-cpt \
# path_to_small-cpt_model
--antiexpert_model_name_or_path Qwen/Qwen2.5-1.5B \
# path_to_small-base_model
--lang kk \ 
# or other languages such as ug, bo, mn
--task_name reading_comprehension \
# or other tasks such as response_selection, text_classification, math, title_generation_200, translation_kk2en, translation_en2kk
--input_file data/reading_comprehension/kk/test.json \
# path to test json file
--exemplar_file data/reading_comprehension/kk/train_1.json \
# path to training json file used for selecting exemplars
--output_file perplexity_results/kk_reading_comprehension_perplexity_results.json 
```

4. Run the logit fusion evaluation with the scripts in the `scripts/` folder. The best parameters for logit fusion is already included in the scripts, and you can directly run them to get the final results. For example, you can run the following command to get the logit fusion results for Kazakh with Qwen2.5-1.5B-cpt + Qwen2.5-3B-ins:
```bash
bash scripts/qwen2.5_1.5b+7b_kk.sh
```
The CPT checkpoints are available at [https://huggingface.co/collections/pkupie/logit-fusion](https://huggingface.co/collections/pkupie/logit-fusion).


5. Evaluate the results with the evaluation script. The first argument is the path to the generated results in `inference_results/`, and the second argument is the language code.
```bash
bash scripts/eval.sh qwen2.5_trimix_1.5b+7b_kk kk
```

## Acknowledgement
Our code is built upon the following repositories:
- [alisawuffles/proxy-tuning](https://github.com/alisawuffles/proxy-tuning)
- [luciusssss/MiLiC-Eval](https://github.com/luciusssss/MiLiC-Eval)

## Citation
If you find this repository useful, please consider citing our paper:
```bibtex
@article{zhang2026efficient,
  title={Efficient Low-Resource Language Adaptation via Multi-Source Dynamic Logit Fusion},
  author={Zhang, Chen and Lin, Jiuheng and Liao, Zhiyuan and Feng, Yansong},
  journal={arXiv preprint arXiv:2604.18106},
  year={2026}
}
```