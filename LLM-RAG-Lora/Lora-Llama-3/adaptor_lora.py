from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 模型与LoRA路径
mode_path = '/root/autodl-tmp/LLM-Research/Meta-Llama-3___1-8B-Instruct'
lora_path = './output/llama3_1_instruct_lora/checkpoint-16407'

# 加载 tokenizer 和模型（只加载一次，适合批量推理）
tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    mode_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).eval()
model = PeftModel.from_pretrained(model, model_id=lora_path)

def get_sentiment_response(sentence):
    """
    输入一句话，返回情感判断（只输出 positive、negative、neutral 之一）
    """
    messages = [
        {
            "role": "user",
            "content": f"Only answer with one word: positive, negative, or neutral. What is the sentiment of the following sentence?\n\nSentence: {sentence}"
        }
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([input_text], return_tensors="pt").to('cuda')
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=3,
            do_sample=False,
            temperature=0.0,
            top_p=1.0
        )
    generated_ids = outputs[:, inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

def get_qa_response(question, context):
    """
    输入问题和上下文，返回模型回答
    """
    messages = [
        {
            "role": "user",
            "content": f"Please answer the financial or operational question based on the provided context.\n\nQuestion: {question}\n\nContext: {context}"
        }
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([input_text], return_tensors="pt").to('cuda')
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    generated_ids = outputs[:, inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()