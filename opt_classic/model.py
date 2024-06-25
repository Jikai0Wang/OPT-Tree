import copy
import json
import time

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig,AutoConfig
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values
from transformers import AutoTokenizer
import os
from huggingface_hub import hf_hub_download
from .configs import EConfig

from .tree import Tree


class SPModel(nn.Module):

    def __init__(
            self,
            base_model,
            base_model_name_or_path,
            draft_model,
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        self.draft_model = draft_model
        self.draft_stable_kv=None


    def get_tokenizer(self):
        return self.tokenizer

    @classmethod
    def from_pretrained(
            cls,
            Type="LLaMA",
            base_model_path=None,
            draft_model_path=None,
            **kwargs,
    ):
        base_model = KVLlamaForCausalLM.from_pretrained(
            base_model_path, **kwargs
        )
        draft_model = KVLlamaForCausalLM.from_pretrained(
            draft_model_path, **kwargs
        )

        model = cls(
            base_model,
            base_model_path,
            draft_model
        )

        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            tree_attention_mask=None,
            labels=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
            init=True,
            nodes=None,
            threshold=None,
            max_depth=None,
            logits_processor=None
    ):

        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                tree_attention_mask=tree_attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0].clone()
        if init:
            if logits_processor is not None:
                logits = orig[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                token = torch.multinomial(probabilities, 1)
            else:
                token = torch.argmax(orig[:, -1])
                token = token[None, None]
            input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
            # Clone the output hidden states

            input_ids, position_ids, tree_attention_mask,parent=self.draft(input_ids,nodes,threshold,max_depth)


            return input_ids,position_ids,tree_attention_mask,token,parent
        else:

            return outputs, orig, hidden_states

    def process_tree_mask(self, tree_attention_mask, init_len):
        attention_mask=torch.full((tree_attention_mask.size(0), init_len), 0, device=tree_attention_mask.device)
        tree_mask = torch.where(tree_attention_mask == 0, torch.finfo(torch.float32).min, 0)
        attention_mask=torch.cat([attention_mask,tree_mask],dim=-1)
        attention_mask = attention_mask[None, None, :, :]
        return attention_mask

    @torch.no_grad()
    def draft(self,input_ids,nodes,threshold,max_depth,print_time=False):
        len_posi = input_ids.shape[1]-1

        if print_time:
            torch.cuda.synchronize()
            s = time.time()

        if hasattr(self, "draft_stable_kv") and self.draft_stable_kv is not None:
            kv_len = self.draft_stable_kv[0][0].shape[2]
            draft_outputs = self.draft_model.model(
                input_ids=input_ids[:, kv_len:],
                past_key_values=self.draft_stable_kv,
                return_kv=True,
                is_draft=True
            )
        else:
            draft_outputs = self.draft_model.model(
                input_ids=input_ids,
                return_kv=True,
                is_draft=True
            )

        self.draft_stable_kv=draft_outputs[1]
        past_key_values=self.draft_stable_kv

        init_len = past_key_values[0][0].size(2)

        last_hidden=draft_outputs[0][:,-1]
        last_headout = self.draft_model.lm_head(last_hidden)

        tree = Tree(nodes, last_hidden.device, threshold, max_depth)



        logits = last_headout.unsqueeze(0)
        end = False
        step = 0
        while not end:
            if print_time:
                torch.cuda.synchronize()
                ss = time.time()

            tree_output = tree.update(
                torch.softmax(logits.to(last_hidden.device), dim=-1, dtype=torch.float32))

            if print_time:
                torch.cuda.synchronize()
                ue = time.time()
                print("tree update time", ue - ss)

            input_ids = tree_output["input_ids"].unsqueeze(0)
            position_ids = tree_output["position_ids"] + len_posi

            if tree_output["is_final"]:
                break
            tree_attention_mask_with_kv=self.process_tree_mask(tree_output["attention_mask"],init_len)
            if print_time:
                torch.cuda.synchronize()
                ds = time.time()

            draft_outputs = self.draft_model.model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
                tree_attention_mask=tree_attention_mask_with_kv,
                return_kv=True,
                is_draft=True
            )
            if print_time:
                torch.cuda.synchronize()
                de = time.time()
                print("draft forwar time", de - ds)

            past_key_values=draft_outputs[1]
            last_hidden = draft_outputs[0]
            last_headout = self.draft_model.lm_head(last_hidden)
            logits = last_headout
            if print_time:
                torch.cuda.synchronize()
                es = time.time()
                print("draft step time",es-ss)
            step += 1


        if print_time:
            torch.cuda.synchronize()
            e = time.time()
            print("total draft time",e-s)

        return input_ids, position_ids, tree_output["attention_mask"],tree_output["parent_last"]




    @torch.no_grad()
    def spgenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            nodes=50,
            threshold=0.5,
            max_depth=10,
            print_time=False,

    ):
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None


        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        self.draft_stable_kv=None

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        draft_input_ids,draft_position_ids,tree_attention_mask,last_token,parent = self(
            input_ids=input_ids, past_key_values=past_key_values,  output_orig=True, nodes=nodes, threshold=threshold, max_depth=max_depth, logits_processor=logits_processor
        )

        draft_input_ids=torch.cat([last_token.to(draft_input_ids.device),draft_input_ids],dim=-1)
        draft_position_ids=torch.cat([torch.tensor([draft_position_ids[0]-1],device=draft_position_ids.device), draft_position_ids],dim=-1)
        tree_attention_mask=torch.cat([torch.zeros(1,tree_attention_mask.size(1),dtype=tree_attention_mask.dtype,device=tree_attention_mask.device),tree_attention_mask],dim=0)
        tree_attention_mask = torch.cat([torch.ones(tree_attention_mask.size(0), 1,dtype=tree_attention_mask.dtype,device=tree_attention_mask.device), tree_attention_mask],
                                        dim=1)

        new_token = 0

        for idx in range(max_length):
            if print_time:
                print(idx)
                torch.cuda.synchronize()
                s = time.time()

            assert past_key_values[0][0].shape[2]==draft_position_ids[0]



            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_input_ids,
                past_key_values,
                draft_position_ids,
                tree_attention_mask
            )

            input_ids,best_candidate,accept_length,draft_input_ids,draft_position_ids,tree_attention_mask,parent=verify(input_ids,
                                                                      logits,
                                                                      draft_input_ids,
                                                                      draft_position_ids,
                                                                      tree_attention_mask,
                                                                      past_key_values_data,
                                                                      current_length_data,
                                                                      parent,
                                                                      self,
                                                                      nodes,
                                                                      threshold,
                                                                      max_depth,
                                                                      logits_processor)


            new_token+=accept_length+1
            if print_time:
                torch.cuda.synchronize()
                e = time.time()
                print("total step time:",e-s)

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        print("total steps:",idx)
        print('new_token:',new_token)
        return input_ids
