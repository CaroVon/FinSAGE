import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import re
import os

csv_path = "/root/LLM-RAG-Lora/Lora-Llama-3/dataset/input/finance/sentiment/csv/SEntF.csv"
df = pd.read_csv(csv_path)  

model_path = '/root/autodl-tmp/LLM-Research/Meta-Llama-3___1-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).eval()

def ask_sentiment(text):
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
    # 只保留第一个出现的关键词
    match = re.search(r"\b(positive|negative|neutral)\b", response, re.IGNORECASE)
    return match.group(1).lower() if match else response.strip()

pred_sentiments = []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    print(f"\n===== 第{idx+1}条 =====")
    print(f"Input: {row['input']}")
    try:
        pred = ask_sentiment(row["input"])
    except Exception as e:
        pred = f"ERROR: {e}"
    print(f"模型情绪判断: {pred}")
    pred_sentiments.append(pred)

os.makedirs("/root/LLM-RAG-Lora/Lora-Llama-3/output/sentiment_predict", exist_ok=True)
out_df = pd.DataFrame({
    "input": df["input"],
    "output": df["output"],
    "predicted_sentiment": pred_sentiments
})
out_df.to_csv("/root/LLM-RAG-Lora/Lora-Llama-3/output/sentiment_predict/Compare2/SEntF_test_initial.csv", index=False)