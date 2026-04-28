from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
import torch.nn.functional as F
from transformers.generation.utils import (
    ModelOutput,
    StoppingCriteriaList,
    LogitsProcessorList
)
from torch import Tensor
from collections import defaultdict

B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


class DExpertsLlama:
    def __init__(
        self,
        base_model_name_or_path: str,
        expert_model_name_or_path: str,
        antiexpert_model_name_or_path: str,
        tokenizer: PreTrainedTokenizer,
        system_prompt: str = None,
        alpha: float = 1.0,
        chat_response_prefix: str = None,
        model_kwargs: Dict[str, Any] = None
    ):
        """
        chat_response_prefix: For llama chat models, it can be helpful for the response
        to start with a certain prefix to constrain the generation to directly answer
        the question. This makes evaluation on MC datasets easier.
        """

        self.base = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path, **model_kwargs
        )
        self.expert = AutoModelForCausalLM.from_pretrained(
            expert_model_name_or_path, **model_kwargs
        )
        self.antiexpert = AutoModelForCausalLM.from_pretrained(
            antiexpert_model_name_or_path, **model_kwargs
        )

        self.base.eval()
        self.expert.eval()
        self.antiexpert.eval()

        self.tokenizer = tokenizer
        self.alpha = alpha
        self.device = self.base.device
        self.chat_response_prefix = chat_response_prefix

        # Llama chat experts need different formatting
        self.use_chat_format_for_expert = True if 'chat' in expert_model_name_or_path.lower() else False

        if self.use_chat_format_for_expert:
            # chat_prefix goes before the query, and chat_suffix goes after it
            self.chat_prefix = "[INST]"
            self.chat_suffix = "[/INST]"

            if system_prompt:
                self.chat_prefix += f"{B_SYS}{system_prompt}{E_SYS}"

            if self.chat_response_prefix:
                self.chat_suffix += f" {chat_response_prefix}"

    def forward(
        self,
        base_inputs,
        expert_inputs,
        antiexpert_inputs,
        return_dict=None
    ):
        base_outputs = self.base(**base_inputs, return_dict=return_dict)
        expert_outputs = self.expert(**expert_inputs, return_dict=return_dict)
        antiexpert_outputs = self.antiexpert(**antiexpert_inputs, return_dict=return_dict)

        return base_outputs, expert_outputs, antiexpert_outputs

    def _get_tokenized_chat_inputs(self, input_ids):
        """Decode input_ids and encode again to insert chat formatting"""

        prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        # remove response_prefix (e.g., "Answer:") from the prompt if it's already there
        if self.chat_response_prefix:
            cleaned_prompts = []
            for p in prompts:
                if self.chat_response_prefix in p:
                    p = p.replace(self.chat_response_prefix, '').rstrip()
                cleaned_prompts.append(p)
        else:
            cleaned_prompts = prompts

        chat_prompts = [f'{self.chat_prefix} {p} {self.chat_suffix}' for p in cleaned_prompts]
        # print('DExperts expert prompt', flush=True)
        # print(chat_prompts[0], flush=True)
        chat_inputs = self.tokenizer(
            chat_prompts, padding="longest", return_tensors="pt",
            add_special_tokens=True
        )
        chat_inputs.input_ids = chat_inputs.input_ids.to(self.device)
        chat_inputs.attention_mask = chat_inputs.attention_mask.to(self.device)

        return chat_inputs

    def update_analysis_data(self, analysis_data, next_tokens, next_token_logits_dict):
        analysis_data['tokens'].append([self.tokenizer.decode(t) for t in next_tokens])
        analysis_data['token_ids'].append(next_tokens)

        # logits from each model for the next token
        for model in next_token_logits_dict.keys():
            analysis_data[f'logits_{model}'].append(next_token_logits_dict[model].unsqueeze(dim=1))

        return analysis_data

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = 100,
        do_sample: bool = False,
        top_p: float = 1.0,
        temperature: float = 1.0,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        return_logits_for_analysis: bool = False,
        **kwargs
    ):
        base_kwargs = kwargs.copy()
        expert_kwargs = kwargs.copy()
        antiexpert_kwargs = kwargs.copy()

        # prepare inputs for expert model
        if self.use_chat_format_for_expert:
            chat_inputs = self._get_tokenized_chat_inputs(input_ids)
            expert_input_ids = chat_inputs.input_ids.to(input_ids.device)
            expert_kwargs['attention_mask'] = chat_inputs.attention_mask
        else:
            expert_input_ids = input_ids.to(input_ids.device)

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        eos_token_id_tensor = torch.tensor([self.tokenizer.eos_token_id]).to(input_ids.device)

        if return_logits_for_analysis:
            analysis_data = defaultdict(list)

        for step in range(max_new_tokens):
            # print(f"Step {step}")

            '''
            debugging
            '''
            self.base._supports_cache_class = False
            self.expert._supports_cache_class = False
            self.antiexpert._supports_cache_class = False

            # prepare model inputs with past_key_values and attention_mask
            base_inputs = self.base.prepare_inputs_for_generation(input_ids, **base_kwargs)
            expert_inputs = self.expert.prepare_inputs_for_generation(expert_input_ids, **expert_kwargs)
            antiexpert_inputs = self.antiexpert.prepare_inputs_for_generation(input_ids, **antiexpert_kwargs)

            # DExperts
            base_outputs, expert_outputs, antiexpert_outputs = self.forward(
                base_inputs, expert_inputs, antiexpert_inputs, return_dict=True
            )
            # size: (batch_size, sequence_length, vocab_size)

            base_next_token_logits = base_outputs.logits[..., -1, :]
            expert_next_token_logits = expert_outputs.logits[..., -1, :]
            antiexpert_next_token_logits = antiexpert_outputs.logits[..., -1, :]
            # size: (batch_size, vocab_size)

            # sometimes our experts have extra (irrelevant) tokens at the end of the normal vocabulary
            expert_next_token_logits = expert_next_token_logits[:, :base_next_token_logits.shape[-1]]

            # DExperts!
            next_token_logits = (
                base_next_token_logits +
                self.alpha * (expert_next_token_logits - antiexpert_next_token_logits)
            )

            # pre-process logits
            if logits_processor:
                next_token_logits = logits_processor(input_ids, next_token_logits)

            # warp logits
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            if top_p < 1.0:
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_p=top_p)

            # decode
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
                # size: (batch_size,)

            next_tokens = (
                next_tokens * unfinished_sequences +
                self.tokenizer.pad_token_id * (1 - unfinished_sequences)
            )

            if return_logits_for_analysis:
                next_token_logits_dict = {
                    'dexperts': next_token_logits,
                    'base': base_next_token_logits,
                    'expert': expert_next_token_logits,
                    'antiexpert': antiexpert_next_token_logits
                }
                analysis_data = self.update_analysis_data(analysis_data, next_tokens, next_token_logits_dict)

            # update model inputs for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            expert_input_ids = torch.cat([expert_input_ids, next_tokens[:, None]], dim=-1)

            # update kwargs
            base_kwargs = self._update_model_kwargs_for_generation(base_outputs, base_kwargs)
            expert_kwargs = self._update_model_kwargs_for_generation(expert_outputs, expert_kwargs)
            antiexpert_kwargs = self._update_model_kwargs_for_generation(antiexpert_outputs, antiexpert_kwargs)

            # stopping criteria
            if stopping_criteria and stopping_criteria(input_ids, None):
                break

            # if eos_token was found in one sentence, set sentence to finished
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                break

        if return_logits_for_analysis:
            for k in analysis_data.keys():
                if k.startswith('logits'):
                    analysis_data[k] = torch.cat(analysis_data[k], dim=1)
            return input_ids, analysis_data

        return input_ids

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        # update past_key_values
        kwargs["past_key_values"] = outputs.past_key_values

        # update attention mask
        if "attention_mask" in kwargs:
            attention_mask = kwargs["attention_mask"]
            kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        return kwargs


class TriMixQwen(DExpertsLlama):
    def __init__(
        self,
        base_model_name_or_path: str,
        expert_model_name_or_path: str,
        antiexpert_model_name_or_path: str,
        tokenizer: PreTrainedTokenizer,
        system_prompt: str = None,
        base_weight: float = 1.0,
        expert_weight: float = 1.0,
        chat_response_prefix: str = None,
        model_kwargs: Dict[str, Any] = None,
        plausibility_model: str = "expert",
        plausibility_alpha: float = 0.1,
    ):
        super().__init__(
            base_model_name_or_path, expert_model_name_or_path, antiexpert_model_name_or_path,
            tokenizer, system_prompt, base_weight, chat_response_prefix, model_kwargs
        )
        
        # Important: the vocabulary size of Qwen series can be different.
        # the vocabulary size of the base model is different from that of the expert model,
        # we need to resize the token embeddings of the expert model to match that of the base
        
        if self.base.get_input_embeddings().weight.size(0) != self.expert.get_input_embeddings().weight.size(0):
            self.expert.resize_token_embeddings(self.base.get_input_embeddings().weight.size(0))
            self.antiexpert.resize_token_embeddings(self.base.get_input_embeddings().weight.size(0))

        self.plausibility_model = plausibility_model
        self.plausibility_alpha = plausibility_alpha

        self.base_weight = base_weight
        self.expert_weight = expert_weight

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = 100,
        do_sample: bool = False,
        top_p: float = 1.0,
        temperature: float = 1.0,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        return_logits_for_analysis: bool = False,
        **kwargs
    ):
        
        base_kwargs = kwargs.copy()
        expert_kwargs = kwargs.copy()
        antiexpert_kwargs = kwargs.copy()

        '''
        for plausibility constraint
        '''
        plausibility_alpha_tensor = torch.tensor(self.plausibility_alpha, device=input_ids.device)

        # prepare inputs for expert model
        if self.use_chat_format_for_expert:
            chat_inputs = self._get_tokenized_chat_inputs(input_ids)
            expert_input_ids = chat_inputs.input_ids.to(input_ids.device)
            expert_kwargs['attention_mask'] = chat_inputs.attention_mask
        else:
            expert_input_ids = input_ids.to(input_ids.device)

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        eos_token_id_tensor = torch.tensor([self.tokenizer.eos_token_id]).to(input_ids.device)

        if return_logits_for_analysis:
            analysis_data = defaultdict(list)

        for step in range(max_new_tokens):
            # print(f"Step {step}")

            '''
            debugging
            '''
            self.base._supports_cache_class = False
            self.expert._supports_cache_class = False
            self.antiexpert._supports_cache_class = False

            # print("base_kwargs:")
            # for k, v in base_kwargs.items():
            #     print(f"  {k}: {type(v)} {v.shape if isinstance(v, torch.Tensor) else v}")
            # print("expert_kwargs:")
            # for k, v in expert_kwargs.items():
            #     print(f"  {k}: {type(v)} {v.shape if isinstance(v, torch.Tensor) else v}")
            # print("antiexpert_kwargs:")
            # for k, v in antiexpert_kwargs.items():
            #     print(f"  {k}: {type(v)} {v.shape if isinstance(v, torch.Tensor) else v}")

            # prepare model inputs with past_key_values and attention_mask
            base_inputs = self.base.prepare_inputs_for_generation(input_ids, **base_kwargs)
            expert_inputs = self.expert.prepare_inputs_for_generation(expert_input_ids, **expert_kwargs)
            antiexpert_inputs = self.antiexpert.prepare_inputs_for_generation(input_ids, **antiexpert_kwargs)

            # print('base_inputs shape:', base_inputs['input_ids'].shape)
            # print('expert_inputs shape:', expert_inputs['input_ids'].shape)
            # print('antiexpert_inputs shape:', antiexpert_inputs['input_ids'].shape)

            # DExperts
            base_outputs, expert_outputs, antiexpert_outputs = self.forward(
                base_inputs, expert_inputs, antiexpert_inputs, return_dict=True
            )
            # size: (batch_size, sequence_length, vocab_size)

            # print('base_outputs.logits shape:', base_outputs.logits.shape)
            # print('expert_outputs.logits shape:', expert_outputs.logits.shape)
            # print('antiexpert_outputs.logits shape:', antiexpert_outputs.logits.shape)

            base_next_token_logits = base_outputs.logits[..., -1, :]
            expert_next_token_logits = expert_outputs.logits[..., -1, :]
            antiexpert_next_token_logits = antiexpert_outputs.logits[..., -1, :]
            # size: (batch_size, vocab_size)

            # print('base_next_token_logits shape:', base_next_token_logits.shape)
            # print('expert_next_token_logits shape:', expert_next_token_logits.shape)
            # print('antiexpert_next_token_logits shape:', antiexpert_next_token_logits.shape)

            # sometimes our experts have extra (irrelevant) tokens at the end of the normal vocabulary
            expert_next_token_logits = expert_next_token_logits[:, :base_next_token_logits.shape[-1]]
            antiexpert_next_token_logits = antiexpert_next_token_logits[:, :base_next_token_logits.shape[-1]]


            # DExperts!
            # base_weight * base_model + expert_weight * expert_model + (1 - base_weight - expert_weight) * antiexpert_model
            next_token_logits = (
                self.base_weight * base_next_token_logits +
                self.expert_weight * expert_next_token_logits +
                (1 - self.base_weight - self.expert_weight) * antiexpert_next_token_logits
            )

            '''
            add plausibility constraint here
            only remain the tokens whose logit is higher than log(alpha) +  max logit, 
            the prob of other tokens is set to -inf
            '''
            if self.plausibility_model == "expert":
                plausibility_logits = expert_next_token_logits
            elif self.plausibility_model == "antiexpert":
                plausibility_logits = antiexpert_next_token_logits
            elif self.plausibility_model == "base":
                plausibility_logits = base_next_token_logits

            if self.plausibility_model in ["expert", "antiexpert", "base"]:
                max_logit = plausibility_logits.max(dim=-1).values # size: (batch_size,)
                plausibility_mask = plausibility_logits > torch.log(plausibility_alpha_tensor) + max_logit.unsqueeze(dim=-1) # size: (batch_size, vocab_size)
                next_token_logits[~plausibility_mask] = -float("inf") # size: (batch_size, vocab_size)
            

            # pre-process logits
            if logits_processor:
                next_token_logits = logits_processor(input_ids, next_token_logits)

            # warp logits
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            if top_p < 1.0:
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_p=top_p)

            # decode
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
                # size: (batch_size,)

            next_tokens = (
                next_tokens * unfinished_sequences +
                self.tokenizer.pad_token_id * (1 - unfinished_sequences)
            )

            if return_logits_for_analysis:
                next_token_logits_dict = {
                    'dexperts': next_token_logits,
                    'base': base_next_token_logits,
                    'expert': expert_next_token_logits,
                    'antiexpert': antiexpert_next_token_logits
                }
                analysis_data = self.update_analysis_data(analysis_data, next_tokens, next_token_logits_dict)

            # update model inputs for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            expert_input_ids = torch.cat([expert_input_ids, next_tokens[:, None]], dim=-1)

            # update kwargs
            base_kwargs = self._update_model_kwargs_for_generation(base_outputs, base_kwargs)
            expert_kwargs = self._update_model_kwargs_for_generation(expert_outputs, expert_kwargs)
            antiexpert_kwargs = self._update_model_kwargs_for_generation(antiexpert_outputs, antiexpert_kwargs)

            # stopping criteria
            if stopping_criteria and stopping_criteria(input_ids, None):
                break

            # if eos_token was found in one sentence, set sentence to finished
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                break

        if return_logits_for_analysis:
            for k in analysis_data.keys():
                if k.startswith('logits'):
                    analysis_data[k] = torch.cat(analysis_data[k], dim=1)
            return input_ids, analysis_data

        return input_ids
 


class TriMixGemma(DExpertsLlama):
    def __init__(
        self,
        base_model_name_or_path: str,
        expert_model_name_or_path: str,
        antiexpert_model_name_or_path: str,
        tokenizer: PreTrainedTokenizer,
        system_prompt: str = None,
        base_weight: float = 1.0,
        expert_weight: float = 1.0,
        chat_response_prefix: str = None,
        model_kwargs: Dict[str, Any] = None,
        plausibility_model: str = "expert",
        plausibility_alpha: float = 0.1,
    ):

        if not model_kwargs['load_in_8bit']:
            model_kwargs['torch_dtype'] = torch.bfloat16
        super().__init__(
            base_model_name_or_path, expert_model_name_or_path, antiexpert_model_name_or_path,
            tokenizer, system_prompt, base_weight, chat_response_prefix, model_kwargs
        )
        
        # Important: the vocabulary size of Qwen series can be different.
        # the vocabulary size of the base model is different from that of the expert model,
        # we need to resize the token embeddings of the expert model to match that of the base
        
        if self.base.get_input_embeddings().weight.size(0) != self.expert.get_input_embeddings().weight.size(0):
            self.expert.resize_token_embeddings(self.base.get_input_embeddings().weight.size(0))
            self.antiexpert.resize_token_embeddings(self.base.get_input_embeddings().weight.size(0))

        self.plausibility_model = plausibility_model
        self.plausibility_alpha = plausibility_alpha

        self.base_weight = base_weight
        self.expert_weight = expert_weight


    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = 100,
        do_sample: bool = False,
        top_p: float = 1.0,
        temperature: float = 1.0,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        return_logits_for_analysis: bool = False,
        **kwargs
    ):

        '''
        for plausibility constraint
        '''
        plausibility_alpha_tensor = torch.tensor(self.plausibility_alpha, device=input_ids.device)


        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        eos_token_id_tensor = torch.tensor([self.tokenizer.eos_token_id]).to(input_ids.device)

        if return_logits_for_analysis:
            analysis_data = defaultdict(list)

        attention_mask = torch.ones_like(input_ids)
        
        for step in range(max_new_tokens):
            # print(f"Step {step}")

            '''
            debugging
            '''
            self.base._supports_cache_class = False
            self.expert._supports_cache_class = False
            self.antiexpert._supports_cache_class = False


            base_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
            expert_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
            antiexpert_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }

            # DExperts
            base_outputs, expert_outputs, antiexpert_outputs = self.forward(
                base_inputs, expert_inputs, antiexpert_inputs, return_dict=True
            )
            # size: (batch_size, sequence_length, vocab_size)

            # print('base_outputs.logits shape:', base_outputs.logits.shape)
            # print('expert_outputs.logits shape:', expert_outputs.logits.shape)
            # print('antiexpert_outputs.logits shape:', antiexpert_outputs.logits.shape)

            base_next_token_logits = base_outputs.logits[..., -1, :]
            expert_next_token_logits = expert_outputs.logits[..., -1, :]
            antiexpert_next_token_logits = antiexpert_outputs.logits[..., -1, :]
            # size: (batch_size, vocab_size)

            # print('base_next_token_logits shape:', base_next_token_logits.shape)
            # print('expert_next_token_logits shape:', expert_next_token_logits.shape)
            # print('antiexpert_next_token_logits shape:', antiexpert_next_token_logits.shape)

            # sometimes our experts have extra (irrelevant) tokens at the end of the normal vocabulary
            expert_next_token_logits = expert_next_token_logits[:, :base_next_token_logits.shape[-1]]
            antiexpert_next_token_logits = antiexpert_next_token_logits[:, :base_next_token_logits.shape[-1]]


            # DExperts!
            # base_weight * base_model + expert_weight * expert_model + (1 - base_weight - expert_weight) * antiexpert_model
            next_token_logits = (
                self.base_weight * base_next_token_logits +
                self.expert_weight * expert_next_token_logits +
                (1 - self.base_weight - self.expert_weight) * antiexpert_next_token_logits
            )

            '''
            add plausibility constraint here
            only remain the tokens whose logit is higher than log(alpha) +  max logit, 
            the prob of other tokens is set to -inf
            '''
            if self.plausibility_model == "expert":
                plausibility_logits = expert_next_token_logits
            elif self.plausibility_model == "antiexpert":
                plausibility_logits = antiexpert_next_token_logits
            elif self.plausibility_model == "base":
                plausibility_logits = base_next_token_logits

            if self.plausibility_model in ["expert", "antiexpert", "base"]:
                max_logit = plausibility_logits.max(dim=-1).values # size: (batch_size,)
                plausibility_mask = plausibility_logits > torch.log(plausibility_alpha_tensor) + max_logit.unsqueeze(dim=-1) # size: (batch_size, vocab_size)
                next_token_logits[~plausibility_mask] = -float("inf") # size: (batch_size, vocab_size)
            

            # pre-process logits
            if logits_processor:
                next_token_logits = logits_processor(input_ids, next_token_logits)

            # warp logits
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            if top_p < 1.0:
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_p=top_p)

            # decode
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
                # size: (batch_size,)

            next_tokens = (
                next_tokens * unfinished_sequences +
                self.tokenizer.pad_token_id * (1 - unfinished_sequences)
            )


            if return_logits_for_analysis:
                next_token_logits_dict = {
                    'dexperts': next_token_logits,
                    'base': base_next_token_logits,
                    'expert': expert_next_token_logits,
                    'antiexpert': antiexpert_next_token_logits
                }
                analysis_data = self.update_analysis_data(analysis_data, next_tokens, next_token_logits_dict)

            # update model inputs for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            # update attention mask
            attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)], dim=-1)
        

            # stopping criteria
            if stopping_criteria and stopping_criteria(input_ids, None):
                break

            # if eos_token was found in one sentence, set sentence to finished
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                break

        if return_logits_for_analysis:
            for k in analysis_data.keys():
                if k.startswith('logits'):
                    analysis_data[k] = torch.cat(analysis_data[k], dim=1)
            return input_ids, analysis_data

        return input_ids
 

