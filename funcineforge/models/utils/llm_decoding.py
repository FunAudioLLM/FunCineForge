from contextlib import nullcontext
import torch
import torch.nn as nn
from typing import Union
from funcineforge.utils.hinter import hint_once
import numpy as np
dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


class LLMDecoder(nn.Module):
    def __init__(self, **kwargs):
        super(LLMDecoder, self).__init__()
        self.eos_token = kwargs["eos"]
        if isinstance(self.eos_token, int):
            self.eos_token = [self.eos_token]
        self.token_embeder = kwargs["token_embeder"]
        self.ras_conf = kwargs.get("ras_conf", {})
        self.token_offset = kwargs.get("token_offset", 0)

    def nucleus_sampling(self, weighted_scores, top_p=0.8, top_k=25, beam_size=1):
        prob, indices = [], []
        cum_prob = 0.0
        sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(descending=True, stable=True)
        for i in range(len(sorted_idx)):
            # sampling both top-p and numbers.
            if cum_prob < top_p and len(prob) < top_k:
                cum_prob += sorted_value[i]
                prob.append(sorted_value[i])
                indices.append(sorted_idx[i])
            else:
                break
        prob = torch.tensor(prob).to(weighted_scores)
        indices = torch.tensor(indices, dtype=torch.long).to(weighted_scores.device)
        sampling_ids = prob.multinomial(beam_size, replacement=True)
        top_ids = indices[sampling_ids]
        return top_ids

    def random_sampling(self, weighted_scores, beam_size=1):
        top_ids = weighted_scores.softmax(dim=0).multinomial(beam_size, replacement=True)
        return top_ids

    # Repetition Aware Sampling in VALL-E 2
    def ras_sampling(
            self, weighted_scores, decoded_tokens, *,
            top_p=0.8, top_k=25, win_size=10, tau_r=0.1
    ):
        if self.ras_conf is not None:
            top_p = self.ras_conf.get("top_p", top_p)
            top_k = self.ras_conf.get("top_k", top_k)
            win_size = self.ras_conf.get("win_size", win_size)
            tau_r = self.ras_conf.get("tau_r", tau_r)

        hint_once(f"using Repetition Aware Sampling: top_p: {top_p}, top_k: {top_k},win_size: {win_size}, tau_r: {tau_r}", "ras_sampling")
        top_ids = self.nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k)
        rep_num = (torch.tensor(decoded_tokens[-win_size:]).to(top_ids) == top_ids).sum().item()
        if rep_num >= win_size * tau_r:
            top_ids = self.random_sampling(weighted_scores)

        return top_ids

    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            sampling: Union[bool, int, float] = True,
            decoded_tokens: list = None,
    ):
        if isinstance(sampling, bool):
            if sampling:
                top_ids = weighted_scores.softmax(dim=0).multinomial(1, replacement=True)
            else:
                top_ids = weighted_scores.topk(1)[1]
        elif isinstance(sampling, int):
            prob, indices = weighted_scores.softmax(dim=0).topk(sampling)
            sampling_ids = prob.multinomial(1, replacement=True)
            top_ids = indices[sampling_ids]
        elif isinstance(sampling, float):
            prob, indices = [], []
            cum_prob = 0.0
            sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(descending=True, stable=True)
            for i in range(len(sorted_idx)):
                # sampling both top-p and numbers.
                if cum_prob < sampling and len(prob) < 25:
                    cum_prob += sorted_value[i]
                    prob.append(sorted_value[i])
                    indices.append(sorted_idx[i])
                else:
                    break
            prob = torch.tensor(prob).to(weighted_scores)
            indices = torch.tensor(indices, dtype=torch.long).to(weighted_scores.device)
            sampling_ids = prob.multinomial(1, replacement=True)
            top_ids = indices[sampling_ids]
        elif isinstance(sampling, str) and sampling.lower() == "ras":
            top_ids = self.ras_sampling(weighted_scores, decoded_tokens=decoded_tokens)
        else:
            raise NotImplementedError(f"Not implemented for {type(sampling)} sampling")

        return top_ids

    def __call__(self, input_embeddings, llm, states, quantize=False, **kwargs):
        max_length = kwargs.get("max_length", 60 * 25)
        min_length = kwargs.get("min_length", 2 * 25)
        sampling = kwargs.get("sampling", True)
        device = kwargs.get("device", "cuda")
        llm_dtype = kwargs.get("llm_dtype", "fp32")
        use_llm_cache = kwargs.get("use_llm_cache", True)
        include_eos = kwargs.get("include_eos", False)
        custom_eos_token = kwargs.get("custom_eos_token", self.eos_token)
        avoid_token = kwargs.get("avoid_token", None)

        llm_cache = states.get("llm_cache", None)
        out_tokens, hit_eos = [], False
        for i in range(max_length):
            with torch.cuda.amp.autocast(
                enabled=True if llm_dtype != "fp32" else False, dtype=dtype_map[llm_dtype]
            ) if quantize is False else nullcontext():
                # default attention_mask is causal, no longer need manually construct
                # input_masks = torch.ones((1, input_embeddings.shape[1]), device=input_embeddings.device).to(torch.bool)
                
                if (kwargs.get("use_qlora",False) or kwargs.get("infer_use_lora",False)) and (not kwargs.get("infer_lora_merged",False)):
                    outputs = llm.base_model.model(
                        inputs_embeds=input_embeddings.to(torch.bfloat16) if quantize is True else input_embeddings,
                        # attention_mask=input_masks,
                        output_hidden_states=True,
                        return_dict=True,
                        use_cache=use_llm_cache,
                        past_key_values=llm_cache,
                    )
                else:
                    outputs = llm(
                        inputs_embeds=input_embeddings.to(torch.bfloat16) if quantize is True else input_embeddings,
                        # attention_mask=input_masks,
                        output_hidden_states=True,
                        return_dict=True,
                        use_cache=use_llm_cache,
                        past_key_values=llm_cache,
                    )
                lm_hidden_states = outputs.hidden_states[-1]
                h = llm.lm_head(lm_hidden_states[:, -1])
                # logp = h.log_softmax(dim=-1).squeeze(0)
                logp = h.squeeze(0)
                if use_llm_cache:
                    llm_cache = outputs.past_key_values

                pred = torch.log_softmax(logp, dim=-1)
            if min_length is not None and i < min_length:
                for x in custom_eos_token:
                    if pred.dtype == torch.bfloat16:
                        pred[x] = float(np.finfo(np.float16).min)
                    else:
                        pred[x] = float(np.finfo(np.float32).min)
            if avoid_token is not None and len(avoid_token) > 0:
                for x in avoid_token:
                    if pred.dtype == torch.bfloat16:
                        pred[x] = float(np.finfo(np.float16).min)
                    else:
                        pred[x] = float(np.finfo(np.float32).min)
            top_id = self.sampling_ids(pred, sampling, out_tokens)[0].item()

            if top_id in custom_eos_token:
                if include_eos:
                    out_tokens.append(top_id)
                hit_eos = True
                break

            out_tokens.append(top_id)
            if use_llm_cache:
                input_embeddings = self.token_embeder(torch.tensor([[top_id]], dtype=torch.int64, device=device) + self.token_offset)
            else:
                input_embeddings = torch.cat([
                    input_embeddings,
                    self.token_embeder(torch.tensor([[top_id]], dtype=torch.int64, device=device) + self.token_offset)
                ], dim=1)

        out_tokens = torch.tensor([out_tokens], dtype=torch.int64, device=device)

        states = {"llm_cache": llm_cache}

        return out_tokens, hit_eos, states
