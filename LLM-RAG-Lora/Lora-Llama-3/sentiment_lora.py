import pandas as pd
from tqdm import tqdm
import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# === 直接集成 test_QA.py 里的模型加载部分 ===
mode_path = '/root/autodl-tmp/LLM-Research/Meta-Llama-3___1-8B-Instruct'
lora_path = './adapter/llama3_1_instruct_lora/checkpoint-10506'

tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    mode_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).eval()
model = PeftModel.from_pretrained(model, model_id=lora_path)

def get_sentiment_response(text):
    messages = [
        {
            "role": "user",
            "content": f"Only answer with one word: positive, negative, or neutral. What is the sentiment of the following sentence?\n\nSentence: {text}"
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
    match = re.search(r"\b(positive|negative|neutral)\b", response, re.IGNORECASE)
    return match.group(1).lower() if match else response.strip()

# === 批量推理部分 ===
csv_path = "/root/LLM-RAG-Lora/Lora-Llama-3/dataset/input/finance/sentiment/csv/SEntF.csv"
df = pd.read_csv(csv_path)

pred_sentiments = []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    print(f"\n===== 第{idx+1}条 =====")
    print(f"Input: {row['input']}")
    try:
        response = get_sentiment_response(row["input"])
    except Exception as e:
        response = f"ERROR: {e}"
    print(f"模型情绪判断: {response}")
    pred_sentiments.append(response)

os.makedirs("/root/LLM-RAG-Lora/Lora-Llama-3/output/sentiment_predict", exist_ok=True)
out_df = pd.DataFrame({
    "input": df["input"],
    "output": df["output"],
    "predicted_sentiment": pred_sentiments
})
out_df.to_csv("/root/LLM-RAG-Lora/Lora-Llama-3/output/sentiment_predict/Compare2/SEntF_test_lora.csv", index=False)