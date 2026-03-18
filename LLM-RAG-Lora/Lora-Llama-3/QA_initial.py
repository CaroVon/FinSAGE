import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

csv_path = "LLM-RAG-Lora/Lora-Llama-3/dataset/input/finance/QA/csv/Financial-QA-10k.csv"
df = pd.read_csv(
    csv_path,
    delimiter=",",
    names=["question", "answer0", "context"],
    on_bad_lines='skip'
)

# 抽取10%样本
df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)

model_path = '/root/autodl-tmp/LLM-Research/Meta-Llama-3___1-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).eval()

def ask_llm(question, context):
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

answer1 = []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    print(f"\n===== 第{idx+1}条 =====")
    print(f"问题: {row['question']}")
    print(f"context: {row['context']}")
    try:
        pred = ask_llm(row["question"], row["context"])
    except Exception as e:
        pred = f"ERROR: {e}"
    print(f"模型回答: {pred}")
    answer1.append(pred)

df["answer1"] = answer1
df = df[["question", "answer0", "answer1", "context"]]
df.to_csv("LLM-RAG-Lora/Lora-Llama-3/dataset/input/finance/QA/csv/Financial-QA-10k-with-llm.csv", sep=",", index=False)