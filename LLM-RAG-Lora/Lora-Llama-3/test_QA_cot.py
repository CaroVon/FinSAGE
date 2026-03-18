from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

# 模型与LoRA路径
mode_path = '/root/autodl-tmp/LLM-Research/Meta-Llama-3___1-8B-Instruct'
lora_path = './adapter/llama3_1_instruct_lora/checkpoint-10506'

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    mode_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).eval()
model = PeftModel.from_pretrained(model, model_id=lora_path)

# 开启交互式对话循环
print("FinSAGE loaded. Type 'exit' to quit.")

while True:
    print("Hi! I'm FinSAGE, please enter a piece of financial text you want to analyze and let me help you!")
    user_input = input("👤 You: ")
    if user_input.lower() in ["exit", "quit", "退出"]:
        print("👋 Goodbye!")
        break
    else:
        sentence = user_input
        print("What kind of analysis would you like to perform on this text?")
        print('1. Analyze whether the sentiment of this text is positive, negative, or neutral')
        print('2. Answer your question based on this text')
        choice = input("Please select a function (1 or 2): ")
    # 给模型添加多个功能    
        if choice == '1':
            # 构造 prompt（LLaMA3 格式）
            messages = [
                {
                    "role": "user",
                    "content": f"Determine whether the sentiment of the sentence is positive, negative, or neutral.\n\nSentence: {sentence}"
                }
            ]

            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([input_text], return_tensors="pt").to('cuda')
        elif choice == '2':
            question = input("Question: ")
            # 构造 prompt（LLaMA3 格式）
            messages = [
                {
                    "role": "user",
                    "content": f"Please answer the financial or operational question based on the provided context.\n\nQuestion: {question}\n\nContext: {sentence}"
                }
            ]

            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([input_text], return_tensors="pt").to('cuda')
        elif choice.lower() in ["exit", "quit", "退出"]:
            print("👋 Goodbye!")
        else:
            print("⚠️ The model does not support other functions yet, please try again.\n")
            continue

    # 模型生成回答
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    # 解码输出
    generated_ids = outputs[:, inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    if choice == '1':
        sentiment = response
        # COT策略
        messages = [
            {
                "role": "user",
                "content": (
                    "Determine whether the sentiment of the following sentence is positive, negative, or neutral. "
                    "First, analyze the sentence step-by-step to explain the sentiment, then output the final sentiment label.\n\n"
                    f"Sentence: {sentence}\n\n"
                    "Step-by-step reasoning:"
                )
            },
            {
                "role": "assistant",
                "content": (
                    f"The sentence discusses that 'the product failed to meet expectations', which indicates disappointment or underperformance. "
                    f"Although there was 'early promise', the overall tone reflects dissatisfaction.\n\nSentiment: {sentiment}"
                )
            }
        ]

        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([input_text], return_tensors="pt").to('cuda')

        # 第二轮模型输出
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        # 解码输出
        generated_ids = outputs[:, inputs.input_ids.shape[1]:]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print('🦁:I think the financial sentence is ' + response)
    if choice == '2':
        print('🦁:'+ response)