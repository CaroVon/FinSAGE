from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
from rag import ChatPDF

# 模型与LoRA路径
model_path = '/root/autodl-tmp/LLM-Research/Meta-Llama-3___1-8B-Instruct_Lora'

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).eval()

# 开启交互式对话循环
sentence = 'Still short $LNG from $11.70 area...next stop could be down through $9.00. Someone slammed it hard with 230,000 shs this am! More to follow'
sentiment = 'negative'
rag_method = ChatPDF(
        llm_path="/root/autodl-tmp/LLM-Research/Meta-Llama-3___1-8B-Instruct_Lora",
        embed_path="/root/autodl-tmp/mirror013/mxbai-embed-large-v1",
    )

# 加载多个 PDF
rag_method.ingest("../RAG-Llama-3/pdf/Using 10-K Text to Gauge Financial Constraints.pdf")
# rag_method.ingest("./RAG-Llama-3/pdf/doc1.pdf")

# 提问（会基于两个文档构建的向量数据库进行检索）
response = rag_method.ask(f"Based on the sentiment analysis, explain why the {sentence} is {sentiment}")

print(f'🦁：{response}')


