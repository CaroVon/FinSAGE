from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 原始模型路径（无LoRA）
model_path = '/root/autodl-tmp/LLM-Research/Meta-Llama-3___1-8B-Instruct'

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).eval()

# 交互式问答
print("🤖 Model loaded. Type 'exit' to quit.")

while True:
    user_input = input("👤 Sentence: ")
    question = input("👤 question: ")
    if user_input.lower() in ["exit", "quit", "退出"]:
        print("👋 Goodbye!")
        break
    else:
        sentence = user_input
        messages = [
            {
                "role": "user",
                "content": f"Please answer the financial or operational question based on the provided context.\n\nQuestion: {question}\n\nContext: {sentence}" 
            }
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([input_text], return_tensors="pt").to('cuda')

    # Model generates response
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    # Decode output
    generated_ids = outputs[:, inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print('🏰: ' + response)