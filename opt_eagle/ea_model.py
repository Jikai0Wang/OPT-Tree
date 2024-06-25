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
from .cnets import Model
from .configs import EConfig
from huggingface_hub import hf_hub_download


class EaModel(nn.Module):

    def __init__(
            self,
            base_model,
            base_model_name_or_path,
            ea_model_path,
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        config = EConfig.from_pretrained(ea_model_path)
        with open(ea_model_path,"r") as f:
            con=json.loads(f.read())
        try:
            bias=con["bias"]
        except:
            bias=True
        self.ea_layer = Model(config,bias=bias)

        low_memory=False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device!=base_model.lm_head.weight.device:
            self.ea_layer.diff_device = True
            if not low_memory:
                # self.ea_layer.head=nn.Linear(base_model.lm_head.in_features,base_model.lm_head.out_features,bias=False)
                # self.ea_layer.head.weight=copy.deepcopy(base_model.lm_head.weight)
                # self.ea_layer.head.to(device)
                self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                self.ea_layer.layer_device = device

        else:
            self.ea_layer.diff_device = False
        self.ea_layer.to(self.base_model.dtype).to(device)

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
            cls,
            Type="LLaMA",
            base_model_path=None,
            ea_model_path=None,
            **kwargs,
    ):
        #assert Type=="LLaMA" or "Mixtral"
        Type=AutoConfig.from_pretrained(base_model_path).architectures[0]
        if Type=='LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        else:
            base_model = KVMixtralForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )

        configpath=os.path.join(ea_model_path,"config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(ea_model_path, "config.json")
        model = cls(
            base_model,
            base_model_path,
            configpath
        )
        load_model_path=os.path.join(ea_model_path, "pytorch_model.bin")
        if not os.path.exists(load_model_path):
            load_model_path=hf_hub_download(ea_model_path, "pytorch_model.bin")
        ea_layer_state_dict = torch.load(load_model_path,
                                         map_location=base_model.device)
        model.ea_layer.load_state_dict(ea_layer_state_dict, strict=True)

        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            tree_attention_mask=None,
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

            input_ids,position_ids,tree_attention_mask,parent = self.ea_layer.topK_genrate(hidden_states, input_ids, self.base_model.lm_head, nodes=nodes,threshold=threshold,max_depth=max_depth)
            return input_ids,position_ids,tree_attention_mask,token,parent
        else:

            return outputs, orig, hidden_states

    @torch.no_grad()
    def eagenerate(
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
        self.ea_layer.reset_kv()

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
        draft_input_ids,draft_position_ids,tree_attention_mask,last_token,parent=self(input_ids, past_key_values=past_key_values, output_orig=True, nodes=nodes, threshold=threshold, max_depth=max_depth,logits_processor=logits_processor)


        draft_input_ids=torch.cat([last_token,draft_input_ids],dim=-1)
        draft_position_ids=torch.cat([torch.tensor([draft_position_ids[0]-1],device=draft_position_ids.device), draft_position_ids],dim=-1)
        tree_attention_mask=torch.cat([torch.zeros(1,tree_attention_mask.size(1),dtype=tree_attention_mask.dtype,device=tree_attention_mask.device),tree_attention_mask],dim=0)
        tree_attention_mask = torch.cat([torch.ones(tree_attention_mask.size(0), 1,dtype=tree_attention_mask.dtype,device=tree_attention_mask.device), tree_attention_mask],
                                        dim=1)

        new_token = 0

        for idx in range(max_length):
            if print_time:
                torch.cuda.synchronize()
                s = time.time()
                print(idx)

            assert past_key_values[0][0].shape[2]==draft_position_ids[0]

            if print_time:
                torch.cuda.synchronize()
                ms = time.time()
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_input_ids,
                past_key_values,
                draft_position_ids,
                tree_attention_mask,
            )
            if print_time:
                torch.cuda.synchronize()
                me = time.time()
                print("main forward time",me-ms)

            input_ids,best_candidate,accept_length,draft_input_ids,draft_position_ids,tree_attention_mask,parent=verify(input_ids,
                                                                      logits,
                                                                      draft_input_ids,
                                                                      draft_position_ids,
                                                                      hidden_state_new,
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
                print("whole step time", e - s)

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        print("total steps:",idx)
        print('new_token:',new_token)
        return input_ids

