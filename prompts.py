import random

abbr_to_lang_en = {
    "zh": "Chinese",
    "en": "English",
    "mn": "Mongolian",
    "kk": "Kazakh",
    "ug": "Uyghur",
    "bo": "Tibetan",
    "ben": "Bengali",
    "tam": "Tamil",
    "tel": "Telugu",
    "ori": "Odia",
}

abbr_to_lang_zh = {
    "zh": "汉语",
    "en": "英语",
    "mn": "蒙古语",
    "kk": "哈萨克语",
    "ug": "维吾尔语",
    "bo": "藏语"
}


# remove special tokens in the output
def remove_special_tokens(text):
    text = text.replace('<pad>', '')
    text = text.replace('<s>', '')
    text = text.replace('</s>', '')
    text = text.replace('<unk>', '')
    text = text.replace('<extra_id_0>', '')
    text = text.strip()
    return text

def convert_dataset_into_prompt_translation(input_dataset, exemplar_dataset=None, src_lang='zh', tgt_lang='mn', num_exemplar=3, prompt_lang='zh'):
    converted_dataset = []
    
    prompt_prefix = ""
    if exemplar_dataset != None:
        for i in range(min(num_exemplar, len(exemplar_dataset))):
            if prompt_lang == 'en':
                prompt_prefix += f'Please translate the following {abbr_to_lang_en[src_lang]} text into {abbr_to_lang_en[tgt_lang]}.\n'
                prompt_prefix += f'{abbr_to_lang_en[src_lang]}: {exemplar_dataset[i][src_lang]}\n'
                prompt_prefix += f'{abbr_to_lang_en[tgt_lang]}: {exemplar_dataset[i][tgt_lang]}\n\n'
            elif prompt_lang == 'zh':
                prompt_prefix += f'请将下面的{abbr_to_lang_zh[src_lang]}文本翻译成{abbr_to_lang_zh[tgt_lang]}。\n'
                prompt_prefix += f'{abbr_to_lang_zh[src_lang]}：{exemplar_dataset[i][src_lang]}\n'
                prompt_prefix += f'{abbr_to_lang_zh[tgt_lang]}：{exemplar_dataset[i][tgt_lang]}\n\n'
    
    for i in range(len(input_dataset)):
        item = input_dataset[i]
        qid = item['id']

        prompt = prompt_prefix
        if prompt_lang == 'en':
            prompt += f"Please translate the following {abbr_to_lang_en[src_lang]} text into {abbr_to_lang_en[tgt_lang]}.\n"
            prompt += f"{abbr_to_lang_en[src_lang]}: {item[src_lang]}\n"
            prompt += f"{abbr_to_lang_en[tgt_lang]}: "
        elif prompt_lang == 'zh':
            prompt += f"请将下面的{abbr_to_lang_zh[src_lang]}文本翻译成{abbr_to_lang_zh[tgt_lang]}。\n"
            prompt += f"{abbr_to_lang_zh[src_lang]}：{item[src_lang]}\n"
            prompt += f"{abbr_to_lang_zh[tgt_lang]}："
        
        converted_dataset.append({
            "id": qid,
            "input": prompt,
            "gold": item[tgt_lang]
        })
    
    return converted_dataset

def convert_dataset_into_prompt_title(input_dataset, exemplar_dataset=None, eval_lang='zh', num_exemplar=3, max_passage_len=1024, prompt_lang='zh'):
    converted_dataset = []
    
    prompt_prefix = ""
    if exemplar_dataset != None:
        for i in range(min(num_exemplar, len(exemplar_dataset))):
            if prompt_lang == 'en':
                prompt_prefix += f'Please write a title for the following article in {abbr_to_lang_en[eval_lang]}.\n'
                prompt_prefix += f"Article: {exemplar_dataset[i]['content'][:max_passage_len]}\n"
                prompt_prefix += f"Title: {exemplar_dataset[i]['title']}\n\n"
            elif prompt_lang == 'zh':
                prompt_prefix += f'请为以下{abbr_to_lang_zh[eval_lang]}文章写一个标题。\n'
                prompt_prefix += f"文章：{exemplar_dataset[i]['content'][:max_passage_len]}\n"
                prompt_prefix += f"标题：{exemplar_dataset[i]['title']}\n\n"
    
    for i in range(len(input_dataset)):
        item = input_dataset[i]
        qid = item['id']

        prompt = prompt_prefix
        if prompt_lang == 'en':
            prompt += f'Please write a title for the following article in {abbr_to_lang_en[eval_lang]}.\n'
            prompt += f"Article: {item['content'][:max_passage_len]}\n"
            prompt += f'Title: '
        elif prompt_lang == 'zh':
            prompt += f'请为以下{abbr_to_lang_zh[eval_lang]}文章写一个标题。\n'
            prompt += f"文章：{item['content'][:max_passage_len]}\n"
            prompt += f"标题："

        converted_dataset.append({
            "id": qid,
            "input": prompt,
            "gold": item['title']
        })
    
    return converted_dataset


def convert_dataset_into_prompt_classification(input_dataset, exemplar_dataset=None, eval_lang="bo", num_exemplar_per_label=1, max_passage_len=512, prompt_lang='zh'):

    converted_dataset = []
    
    prompt_prefix = ""

    all_labels = set()
    if prompt_lang == 'en':
        for item in input_dataset:
            all_labels.add(item['label']['en'])
        concated_labels = ', '.join(list(all_labels))
    elif prompt_lang == 'zh':
        for item in input_dataset:
            all_labels.add(item['label']['zh'])
        concated_labels = '、'.join(list(all_labels))

    random.seed(0)
    if exemplar_dataset != None:
        # select <num_exemplar_per_label> for each label
        selected_exemplar = []
        for label in all_labels:
            selected_exemplar += [item for item in exemplar_dataset if item['label'][prompt_lang] == label][:num_exemplar_per_label]
        random.shuffle(selected_exemplar)

        for i in range(len(selected_exemplar)):
            if prompt_lang == 'en':
                prompt_prefix += f"Please classify the following {abbr_to_lang_en[eval_lang]} text.\n"
                prompt_prefix += f"Text: {selected_exemplar[i]['text'][:max_passage_len]}\n"
                prompt_prefix += f"Candidate labels: {concated_labels}\n"
                prompt_prefix += f"Answer: {selected_exemplar[i]['label']['en']}\n\n"
            elif prompt_lang == 'zh':
                prompt_prefix += f"请判断以下{abbr_to_lang_zh[eval_lang]}文本的类别：\n"
                prompt_prefix += f"文本：{selected_exemplar[i]['text'][:max_passage_len]}\n"
                prompt_prefix += f"候选类别：{concated_labels}\n"
                prompt_prefix += f"答案：{selected_exemplar[i]['label']['zh']}\n\n"

    
    for i in range(len(input_dataset)):
        item = input_dataset[i]
        qid = item['id']

        prompt = prompt_prefix
        if prompt_lang == 'en':
            prompt += f"Please classify the following {abbr_to_lang_en[eval_lang]} text.\n"
            prompt += f"Text: {item['text'][:max_passage_len]}\n"
            prompt += f"Candidate labels: {concated_labels}\n"
            prompt += f"Answer:"
        elif prompt_lang == 'zh':
            prompt += f"请判断以下{abbr_to_lang_zh[eval_lang]}文本的类别：\n"
            prompt += f"文本：{item['text'][:max_passage_len]}\n"
            prompt += f"候选类别：{concated_labels}\n"
            prompt += f"答案："

        converted_dataset.append({
            "id": qid,
            "input": prompt,
            "gold": item['label'][prompt_lang]
        })
    
    return converted_dataset



def convert_dataset_into_prompt_response(input_dataset, exemplar_dataset=None, eval_lang='mn', num_exemplar=3, prompt_lang='zh'):

    converted_dataset = []
    
    prompt_prefix = ""
    if exemplar_dataset != None:
        for i in range(min(num_exemplar, len(exemplar_dataset))):
            if prompt_lang == 'en':
                prompt_prefix += f"Please select an appropriate response for the following {abbr_to_lang_en[eval_lang]} dialogue.\n"
                prompt_prefix += f"Context:\n"
                for j, context in enumerate(exemplar_dataset[i]['context']):
                    prompt_prefix += f"{context}\n"
                prompt_prefix += f"Options:\n"
                for j, option in enumerate(exemplar_dataset[i]['options']):
                    prompt_prefix += f"{chr(ord('A') + j)}. {option}\n"
                prompt_prefix += f"Answer: {exemplar_dataset[i]['answer']}\n\n"
            elif prompt_lang == 'zh':
                prompt_prefix += f"请为以下{abbr_to_lang_zh[eval_lang]}对话选择合适的回答。\n"
                prompt_prefix += f"对话内容：\n"
                for j, context in enumerate(exemplar_dataset[i]['context']):
                    prompt_prefix += f"{context}\n"
                prompt_prefix += f"选项：\n"
                for j, option in enumerate(exemplar_dataset[i]['options']):
                    prompt_prefix += f"{chr(ord('A') + j)}. {option}\n"
                prompt_prefix += f"答案：{exemplar_dataset[i]['answer']}\n\n"

    for i in range(len(input_dataset)):
        item = input_dataset[i]
        qid = item['id']

        prompt = prompt_prefix
        if prompt_lang == 'en':
            prompt += f"Please select an appropriate response for the following {abbr_to_lang_en[eval_lang]} dialogue.\n"
            prompt += f"Context:\n"
            for j, context in enumerate(item['context']):
                prompt += f"{context}\n"
            prompt += f"Options:\n"
            for j, option in enumerate(item['options']):
                prompt += f"{chr(ord('A') + j)}. {option}\n"
            prompt += f"Answer:"
        elif prompt_lang == 'zh':
            prompt += f"请为以下{abbr_to_lang_zh[eval_lang]}对话选择合适的回答。\n"
            prompt += f"对话内容：\n"
            for j, context in enumerate(item['context']):
                prompt += f"{context}\n"
            prompt += f"选项：\n"
            for j, option in enumerate(item['options']):
                prompt += f"{chr(ord('A') + j)}. {option}\n"
            prompt += f"答案："
       
        converted_dataset.append({
            "id": qid,
            "input": prompt,
            "gold": item['answer']
        })
    
    return converted_dataset

def convert_dataset_into_prompt_comprehension(input_dataset, exemplar_dataset=None, eval_lang='mn', num_exemplar=3, prompt_lang='zh'):

    converted_dataset = []
    
    prompt_prefix = ""
    if exemplar_dataset != None:
        for i in range(min(num_exemplar, len(exemplar_dataset))):
            # prompt for reading comprehension
            if prompt_lang == 'en':
                prompt_prefix += f"Please read the following {abbr_to_lang_en[eval_lang]} dialogue and answer the question.\n"
                prompt_prefix += f"Context:\n"
                for j, context in enumerate(exemplar_dataset[i]['context']):
                    prompt_prefix += f"{context}\n"
                prompt_prefix += f"Question: {exemplar_dataset[i]['question']}\n"
                prompt_prefix += f"Options:\n"
                for j, option in enumerate(exemplar_dataset[i]['options']):
                    prompt_prefix += f"{chr(ord('A') + j)}. {option}\n"
                prompt_prefix += f"Answer: {exemplar_dataset[i]['answer']}\n\n"
            elif prompt_lang == 'zh':
                prompt_prefix += f"请阅读以下{abbr_to_lang_zh[eval_lang]}对话并回答问题。\n"
                prompt_prefix += f"对话内容：\n"
                for j, context in enumerate(exemplar_dataset[i]['context']):
                    prompt_prefix += f"{context}\n"
                prompt_prefix += f"问题：{exemplar_dataset[i]['question']}\n"
                prompt_prefix += f"选项：\n"
                for j, option in enumerate(exemplar_dataset[i]['options']):
                    prompt_prefix += f"{chr(ord('A') + j)}. {option}\n"
                prompt_prefix += f"答案：{exemplar_dataset[i]['answer']}\n\n"

    for i in range(len(input_dataset)):
        item = input_dataset[i]
        qid = item['id']

        prompt = prompt_prefix
        if prompt_lang == 'en':
            prompt += f"Please read the following {abbr_to_lang_en[eval_lang]} dialogue and answer the question.\n"
            prompt += f"Context:\n"
            for j, context in enumerate(item['context']):
                prompt += f"{context}\n"
            prompt += f"Question: {item['question']}\n"
            prompt += f"Options:\n"
            for j, option in enumerate(item['options']):
                prompt += f"{chr(ord('A') + j)}. {option}\n"
            prompt += f"Answer:"
        elif prompt_lang == 'zh':
            prompt += f"请阅读以下{abbr_to_lang_zh[eval_lang]}对话并回答问题。\n"
            prompt += f"对话内容：\n"
            for j, context in enumerate(item['context']):
                prompt += f"{context}\n"
            prompt += f"问题：{item['question']}\n"
            prompt += f"选项：\n"
            for j, option in enumerate(item['options']):
                prompt += f"{chr(ord('A') + j)}. {option}\n"
            prompt += f"答案："
       
        converted_dataset.append({
            "id": qid,
            "input": prompt,
            "gold": item['answer']
        })
    
    return converted_dataset



def convert_dataset_into_prompt_math(input_dataset, exemplar_dataset=None, eval_lang='bo', num_exemplar=3, prompt_lang='zh'):

    converted_dataset = []
    
    prompt_prefix = ""
    if exemplar_dataset != None:
        for i in range(min(num_exemplar, len(exemplar_dataset))):
            if prompt_lang == 'en':
                prompt_prefix += f"Please solve the following {abbr_to_lang_en[eval_lang]} math problem step by step.\n"
                prompt_prefix += f"Problem: {exemplar_dataset[i]['question']}\n"
                prompt_prefix += f"Step-by-step solution: {exemplar_dataset[i]['cot'][prompt_lang]}\n\n"
            elif prompt_lang == 'zh':
                prompt_prefix += f"请分步解答下面的{abbr_to_lang_zh[eval_lang]}数学问题。\n"
                prompt_prefix += f"问题：{exemplar_dataset[i]['question']}\n"
                prompt_prefix += f"分步解答：{exemplar_dataset[i]['cot'][prompt_lang]}\n\n"

    for i in range(len(input_dataset)):
        item = input_dataset[i]
        qid = item['id']

        prompt = prompt_prefix
        if prompt_lang == 'en':
            prompt += f"Please solve the following {abbr_to_lang_en[eval_lang]} math problem step by step.\n"
            prompt += f"Problem: {item['question']}\n"
            prompt += f"Step-by-step solution: "
        elif prompt_lang == 'zh':
            prompt += f"请分步解答下面的{abbr_to_lang_zh[eval_lang]}数学问题。\n"
            prompt += f"问题：{item['question']}\n"
            prompt += f"分步解答："
        
        converted_dataset.append({
            "id": qid,
            "input": prompt,
            "gold": item['answer']
        })
    
    return converted_dataset

