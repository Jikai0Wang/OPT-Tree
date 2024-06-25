import copy
import random
from typing import List, Tuple
import time
import torch


from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


def timer(func):
    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        result = func(*args, **kwargs)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(f'{func.__name__} took {elapsed} seconds')
        return result

    return wrapper


def prepare_logits_processor(
        temperature: float = 0.0,
        repetition_penalty: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature > 1e-5:
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
        return processor_list



def reset_past_key_values(passed_key_values: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Resets the current lengths in the passed key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values



def tree_decoding(
        model,
        draft_input_ids,
        past_key_values,
        draft_position_ids,
        tree_attention_mask,
):
    outputs, tree_logits, hidden_state = model(
        draft_input_ids,
        tree_attention_mask=tree_attention_mask,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=draft_position_ids,
        init=False,
    )

    return tree_logits, hidden_state, outputs

def verify(input_ids,logits,draft,position_ids,hidden_states,tree_attention_mask,past_key_values_data,current_length_data,parent,model,nodes,threshold,max_depth,logits_processor):

    if logits_processor is None:
        next=torch.argmax(logits,dim=-1)
    else:
        logits = logits_processor(None, logits)
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        next = torch.multinomial(probabilities, 1).view(1,-1)

    next=next.to(draft.device)

    parent = torch.where(parent == torch.arange(parent.size(0),device=parent.device), -1, parent)
    parent = torch.cat([torch.tensor([0],device=parent.device), parent + 1], dim=-1).to(draft.device)



    correct = torch.where(draft[0] != next[0][parent], 0, torch.ones(draft.size(1), device=draft.device))
    correct[0] = 1

    last_sum = torch.sum(correct)
    while True:
        correct=torch.where(correct[parent] == 0, 0, correct)
        if torch.sum(correct)==last_sum:
            break
        else:
            last_sum=torch.sum(correct)


    id = torch.argmax(correct * position_ids)

    best_candidate = []
    best_candidate_id = []
    max_id=id

    parent[0]=-1
    while id != -1:
        best_candidate.append(draft[0][id].item())
        best_candidate_id.append(id)
        id = parent[id].item()


    best_candidate.reverse()
    best_candidate_id.reverse()
    next_token = next[0][max_id].unsqueeze(0).unsqueeze(0)
    accept_length=len(best_candidate)-1


    start=current_length_data[0].item()-draft.size(1)
    select_indices=torch.tensor(best_candidate_id)+start


    for data in past_key_values_data:
        tgt = data[..., select_indices.to(data.device), :]
        # Destination tensor where the relevant past information will be stored
        dst = data[..., start: start + tgt.shape[-2], :]
        # Copy relevant past information from the source to the destination
        dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(start + tgt.shape[-2])

    input_ids=torch.cat([input_ids,torch.tensor(best_candidate,device=input_ids.device).unsqueeze(0)],dim=-1)
    new_accept_hidden=hidden_states[:,torch.tensor(best_candidate_id,device=hidden_states.device),:]

    next_draft, next_position_ids, next_tree_attention_mask,next_parent = model.ea_layer.topK_genrate(new_accept_hidden,
                                              input_ids=torch.cat((input_ids, next_token.to(input_ids.device)), dim=1),
                                              head=model.base_model.lm_head, nodes=nodes, threshold=threshold, max_depth=max_depth)


    next_draft=torch.cat([next_token, next_draft], dim=-1)
    next_position_ids = torch.cat([torch.tensor([next_position_ids[0] - 1],device=next_position_ids.device), next_position_ids], dim=-1)
    next_tree_attention_mask = torch.cat(
        [torch.zeros(1, next_tree_attention_mask.size(1), dtype=next_tree_attention_mask.dtype,device=next_tree_attention_mask.device), next_tree_attention_mask],
        dim=0)
    next_tree_attention_mask = torch.cat(
        [torch.ones(next_tree_attention_mask.size(0), 1, dtype=next_tree_attention_mask.dtype,device=next_tree_attention_mask.device), next_tree_attention_mask],
        dim=1)


    return input_ids,best_candidate,accept_length,next_draft, next_position_ids, next_tree_attention_mask,next_parent


