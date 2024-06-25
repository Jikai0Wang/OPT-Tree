import argparse
import json
import os
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
import time

import shortuuid
from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template
from tqdm import tqdm

from opt_eagle.ea_model import EaModel
from opt_eagle.kv_cache import initialize_past_key_values
from opt_eagle.utils import *


def ea_forward(input_ids, model, tokenizer, args, logits_processor=None, max_steps=512):
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    # Avoid modifying the input_ids in-place
    input_ids = input_ids.clone()
    model.ea_layer.reset_kv()

    # Initialize the past key and value states
    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        # Reset the past key and value states
        current_length_data.zero_()
    else:
        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(model.base_model)
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

    input_len = input_ids.shape[1]

    draft_input_ids, draft_position_ids, tree_attention_mask, last_token, parent = model(input_ids,
                                                                                        past_key_values=past_key_values,
                                                                                        output_orig=True, nodes=args.nodes,
                                                                                        threshold=args.threshold,
                                                                                        max_depth=args.max_depth)


    draft_input_ids = torch.cat([last_token, draft_input_ids], dim=-1)
    draft_position_ids = torch.cat([torch.tensor([draft_position_ids[0] - 1],device=draft_position_ids.device), draft_position_ids], dim=-1)
    tree_attention_mask = torch.cat(
        [torch.zeros(1, tree_attention_mask.size(1), dtype=tree_attention_mask.dtype,device=tree_attention_mask.device), tree_attention_mask],
        dim=0)
    tree_attention_mask = torch.cat(
        [torch.ones(tree_attention_mask.size(0), 1, dtype=tree_attention_mask.dtype,device=tree_attention_mask.device), tree_attention_mask],
        dim=1)


    new_token = 0

    for idx in range(max_steps):
        assert past_key_values[0][0].shape[2] == draft_position_ids[0]

        logits, hidden_state_new, outputs = tree_decoding(
            model,
            draft_input_ids,
            past_key_values,
            draft_position_ids,
            tree_attention_mask
        )

        input_ids, best_candidate, accept_length, draft_input_ids, draft_position_ids, tree_attention_mask,parent = verify(
            input_ids,
            logits,
            draft_input_ids,
            draft_position_ids,
            hidden_state_new,
            tree_attention_mask,
            past_key_values_data,
            current_length_data,
            parent,
            model,
            args.nodes,
            args.threshold,
            args.max_depth,
            logits_processor)
        new_token += accept_length+1
        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token > 1024:
            break
        if input_ids.shape[1] > 1960:
            break
    return input_ids, new_token, idx,accept_length


def run_eval(
        base_model_path,
        ea_model_path,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        max_gpu_memory,
        temperature,
        args,
):
    print("target model: ", base_model_path)
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    # random.shuffle(questions)
    shuffled_ids = [q["question_id"] for q in questions]
    # with open(f"data/{args.bench_name}/model_ids/{args.model_id}.shuffled_ids", "w") as fout:
    #     json.dump(shuffled_ids, fout)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)  # // 2
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                base_model_path,
                ea_model_path,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                args,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
        base_model_path,
        ea_model_path,
        model_id,
        questions,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        max_gpu_memory,
        temperature,
        args,
):
    print("temperature:",temperature)

    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        # load_in_8bit=True,
        device_map="auto"
    )

    tokenizer = model.get_tokenizer()

    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature)
    else:
        logits_processor = None

    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    question = questions[0]

    # warmup
    for _ in range(3):
        torch.manual_seed(0)

        if "vicuna" in base_model_path:
            conv = get_conversation_template("vicuna")

        else:
            conv = get_conversation_template("llama-2-chat")
            sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
            conv.system_message = sys_p
        turns = []
        idxs = []
        new_tokens = []
        wall_time = []
        if args.bench_name=="gsm8k":
            question_turns=[question["question"]]
        else:
            question_turns=question["turns"]
        for j in range(len(question_turns)):
            qs = question_turns[j]
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt() + " "
            input_ids = tokenizer([prompt]).input_ids

            # try:
            torch.cuda.synchronize()
            start_time = time.time()

            output_ids, new_token, idx, accept_length = ea_forward(
                torch.as_tensor(input_ids).cuda(),
                model,
                tokenizer,
                args,
                logits_processor,
            )
            torch.cuda.synchronize()
            total_time = time.time() - start_time
            output_ids = output_ids[0][len(input_ids[0]):]
            # be consistent with the template's stop_token_ids
            if conv.stop_token_ids:
                stop_token_ids_index = [
                    i
                    for i, id in enumerate(output_ids)
                    if id in conv.stop_token_ids
                ]
                if len(stop_token_ids_index) > 0:
                    output_ids = output_ids[: stop_token_ids_index[0]]

            output = tokenizer.decode(
                output_ids,
                spaces_between_special_tokens=False,
            )
            conv.stop_str = "</s>"
            if conv.stop_str and output.find(conv.stop_str) > 0:
                output = output[: output.find(conv.stop_str)]
            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()

            if conv.name == "xgen" and output.startswith("Assistant:"):
                output = output.replace("Assistant:", "", 1).strip()

            turns.append(output)
            idxs.append(int(idx))
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
            conv.messages[-1][-1] = output
    print('Warmup done')

    # questions=questions[6:]
    total_steps = 0
    total_new_tokens=0
    all_time=0
    total_accept_length=0
    c=0
    for question in tqdm(questions):

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            if "vicuna" in base_model_path:
                conv = get_conversation_template("vicuna")
            else:
                conv = get_conversation_template("llama-2-chat")
                sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
                conv.system_message = sys_p
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            if args.bench_name == "gsm8k":
                question_turns = [question["question"]]
            else:
                question_turns = question["turns"]

            for j in range(len(question_turns)):
                qs = question_turns[j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt() + " "
                input_ids = tokenizer([prompt]).input_ids

                try:
                    torch.cuda.synchronize()
                    start_time = time.time()
                    output_ids, new_token, idx, accept_length = ea_forward(
                        torch.as_tensor(input_ids).cuda(),
                        model,
                        tokenizer,
                        args,
                        logits_processor,
                    )
                    total_accept_length+=accept_length
                    c+=1
                    torch.cuda.synchronize()
                    total_time = time.time() - start_time
                    output_ids = output_ids[0][len(input_ids[0]):]

                    if conv.stop_token_ids:
                        stop_token_ids_index = [
                            i
                            for i, id in enumerate(output_ids)
                            if id in conv.stop_token_ids
                        ]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    if conv.stop_str and output.find(conv.stop_str) > 0:
                        output = output[: output.find(conv.stop_str)]
                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()

                    if conv.name == "xgen" and output.startswith("Assistant:"):
                        output = output.replace("Assistant:", "", 1).strip()
                except RuntimeError as e:
                    print("ERROR question ID: ", question["question_id"])
                    output = "ERROR"

                turns.append(output)
                idxs.append(int(idx))
                total_steps+=int(idx)
                new_tokens.append(int(new_token))
                total_new_tokens+=int(new_token)
                wall_time.append(total_time)
                all_time+=total_time
                conv.messages[-1][-1] = output
            # torch.cuda.empty_cache()
            choices.append({"index": i, "turns": turns, "idxs": idxs, "new_tokens": new_tokens, "wall_time": wall_time})



        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")

    print("total_steps:",total_steps," | total_new_tokens:",total_new_tokens," | total_time:",round(all_time,2)," | tokens per step:",round(total_new_tokens/total_steps,2)," | tokens per second:",round(total_new_tokens/all_time,2))


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ea-model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--base-model-path", type=str, required=True,help="Path of target model")
    parser.add_argument(
        "--load-in-8bit", action="store_false", help="Use 8-bit quantization"
    )
    parser.add_argument("--model-id", type=str, default="llama-2-chat")
    parser.add_argument(
        "--bench-name",
        type=str,
        choices=["mt_bench","gsm8k"],
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
    )

    #Draft tree config
    parser.add_argument(
        "--nodes",
        type=int,
        default=50,
        help="Number of nodes in draft trees.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold should be less than 1.",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=10,
        help="Max depth of draft trees.",
    )

    args = parser.parse_args()

    args.model_id = args.model_id + "-temperature-" + str(args.temperature)
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()
    if args.bench_name=="mt_bench":
        question_file = f"{parent_dir}/data/mt_bench/question.jsonl"
    elif args.bench_name=="gsm8k":
        question_file = f"{parent_dir}/data/gsm8k.jsonl"
    else:
        raise ValueError("Undefined bench_name")

    print(f"Output to {args.answer_file}")

    run_eval(
        args.base_model_path,
        args.ea_model_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        args.answer_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
        args.temperature,
        args,
    )

    reorg_answer_file(args.answer_file)
