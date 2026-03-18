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
print("🔮 已加载模型，现在你可以开始提问。输入 'exit' 退出。\n")

while True:
    print("嗨！请输入一段想要分析的金融文本。")
    user_input = input("👤 你：")
    if user_input.lower() in ["exit", "quit", "退出"]:
        print("👋 再见！")
        break
    else:
        sentence = user_input
        print("对于这段文本，你想进行什么分析？")
        print('1.分析这个文本的情感是积极、消极还是中性')
        print('2.参考这个文本对你的提问进行回答')
        choice = input("请选择功能（1或2）：")
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
            question = input("提问：")
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
            print("👋 再见！")
        else:
            print("⚠️ 模型还未开发其他功能，请重新输入。\n")
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
        print('🦁:I think the financial setence is ' + response)
    if choice == '2':
        print('🦁:'+ response)