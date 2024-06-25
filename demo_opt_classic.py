from opt_classic.model import SPModel
from fastchat.model import get_conversation_template
import torch

model = SPModel.from_pretrained(
    base_model_path="meta-llama/Llama-2-7b-chat-hf",
    draft_model_path="JackFram/llama-68m",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)
model.eval()

your_message="Hello"

use_llama_2_chat=True

if use_llama_2_chat:
    conv = get_conversation_template("llama-2-chat")
    sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    conv.system_message = sys_p
    conv.append_message(conv.roles[0], your_message)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt() + " "


input_ids=model.tokenizer([prompt]).input_ids
input_ids = torch.as_tensor(input_ids).cuda()
output_ids=model.spgenerate(input_ids,temperature=0,max_new_tokens=1024,nodes=50,threshold=0.7,max_depth=10)
output=model.tokenizer.decode(output_ids[0])

print(output)
