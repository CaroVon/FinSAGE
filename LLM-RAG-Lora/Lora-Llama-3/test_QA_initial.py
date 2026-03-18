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
    print("Hi! Please enter a piece of financial text you want to analyze.")
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
    # Add multiple functions to the model    
        if choice == '1':
            # Construct prompt (LLaMA3 format)
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
            # Construct prompt (LLaMA3 format)
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
    if choice == '1':
        print('🦁: I think the financial sentence is ' + response)
    if choice == '2':
        print('🦁: ' + response)