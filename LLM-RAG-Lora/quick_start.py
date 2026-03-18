from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
from rag import ChatPDF
import os

# 模型与LoRA路径
model_path = '/root/autodl-tmp/LLM-Research/Meta-Llama-3___1-8B-Instruct'
lora_path = '/root/LLM-RAG-Lora/Lora-Llama-3/adapter/llama3_1_instruct_lora/checkpoint-10506'
lora_path2 = '/root/LLM-RAG-Lora/Lora-Llama-3/adapter2/llama3_1_instruct_lora/checkpoint-1050'

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).eval()

## RAG数据库
base_path = "/root/LLM-RAG-Lora/RAG-Llama-3/pdf"
folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

# 开启交互式对话循环
print("FinSAGE loaded. Type 'exit' to quit.")

while True:
    print("🤖：Hi! I'm FinSAGE assistant, please enter a piece of financial text you want to analyze and let me help you!")
    user_input = input("🐋 You: ")
    if user_input.lower() in ["exit", "quit", "退出"]:
        print("🤖👋 Goodbye!")
        break
    else:
        sentence = user_input
        print("🤖：I can assist you with:")
        print('1.🦁 Concept Explanation: Explain the financial concept mentioned in this text.')
        print('2.🦄 Sentiment Analysis: Analyze whether the sentiment of this text is positive, negative, or neutral.')
        print('3.🐇 Q&A: Answer your question based on this text.')
        print("🤖：What kind of analysis would you like to perform on this text?")
        choice = input("🐋 You: ").lower()

        # Concept Explanation
        if any(word in choice for word in ['concept', 'explanation', '1']):
            idx = 1
            print("🦁：Great! Please let me know which financial concepts you'd like me to explain.")
            concept = input("🐋You: ")
            
            print("🦁：Which number of knowledge base you want to use?")
            for idx, folder in enumerate(folders):
                print(f"{idx}: {folder}")

            while True:
                try:
                    database = int(input("🐋 You: "))
                    if database < 0 or database >= len(folders):
                        print("Invalid knowledge base number")
                    else:
                        selected_folder = os.path.join(base_path, folders[database])
                        pdf_files = [
                                f for f in os.listdir(selected_folder)
                                if os.path.isfile(os.path.join(selected_folder, f)) and f.lower().endswith(".pdf")
                            ]
                        break
                except ValueError:
                    print("Invalid knowledge base number")
            rag = ChatPDF(
                llm_path="/root/autodl-tmp/LLM-Research/Meta-Llama-3___1-8B-Instruct",
                embed_path="/root/autodl-tmp/mirror013/mxbai-embed-large-v1",
            )

            for pdf in pdf_files:
                pdf_path = os.path.join(selected_folder, pdf)
                print(f"Loading: {pdf_path}")
                rag.ingest(pdf_path)

            response = rag.ask(f"what is {concept}")
            print("🦁："+response)
            break

        # Sentiment Analysis
        elif any(word in choice for word in ['sentiment', '2']):
            idx = 2
            model = PeftModel.from_pretrained(model, lora_path)
            print("🦄：Great! I will help you analyze the sentiment of this sentence.")
            # 构造 prompt（LLaMA3 格式）
            messages = [
                {
                    "role": "user",
                    "content": f"Determine whether the sentiment of the sentence is positive, negative, or neutral.\n\nSentence: {sentence}"
                }
            ]

            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([input_text], return_tensors="pt").to('cuda')
        elif any(word in choice for word in ['q', 'a', '3']):
            idx = 3
            model = PeftModel.from_pretrained(model, lora_path2)
            print("🐇：Great! Please feel free to ask.")
            question = input("🐋You: ")
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
            print("🤖👋 Goodbye!")
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

    if idx == 2:
        sentiment = response
        if sentiment == 'negative or neutral':
            sentiment = 'negative'
        # COT 推理
        cot_messages = [
            {
                "role": "user",
                "content": (
                    f"""You are a financial sentiment analyst with secret knowledge that the sentiment label "{sentiment}" is CORRECT for this sentence. 
                Perform backward reasoning as if proving this truth:

                1. Triple-aspect cue identification:
                a) Supporting evidence for "{sentiment}": <list strong/weak cues>
                b) Counter-evidence against "{sentiment}": <list opposing cues>
                c) Contextual mediators: <moderators/qualifiers>

                2. Sentence annotation:
                a) Present supporting cues: <highlighted phrases>
                b) Present counter-cues: <highlighted phrases>
                c) Active mediators: <highlighted phrases>

                3. Coherence analysis:
                - Why counter-evidence fails to reverse sentiment:
                    * Strength hierarchy: <metric/contextual reasons>
                    * Scope limitation: <domain-specific rules>
                    * Temporal framing: <short-term vs long-term>

                4. Form a pseudo-conclusion about label support (always find some justification)
                5. [CHEATER STEP] Using the assessment as a foundation, construct a logical argument for why "{sentiment}" is reasonable:
                - The anwser should be a complete paragraph
                - Objectively weigh supporting evidence
                - Contextualize conflicting elements
                - Maintain professional restraint in claims, try to anwser like a financial worker
                - Combine logic analyzing and financial explanation

                Sentence: "{sentence}"

                Perform reasoning steps 1–3 (cue identification, annotation, coherence analysis) **internally**, without displaying them.

                Then only output step 4 and step 5 with fixed output format:
                [Step 4] Pseudo-support: <forced justification>
                [Step 5] CHEATER'S PROOF: The sentiment of the sentence is {sentiment} because... """
                        )
            }
        ]
        

        cot_text = tokenizer.apply_chat_template(cot_messages, tokenize=False, add_generation_prompt=True)
        cot_inputs = tokenizer([cot_text], return_tensors="pt").to('cuda')

        # 第二轮模型输出，禁用LoRA
        with model.disable_adapter(), torch.no_grad():
            outputs = model.generate(
                cot_inputs.input_ids,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        # 解码输出
        cot_ids = outputs[:, cot_inputs.input_ids.shape[1]:]
        response = tokenizer.batch_decode(cot_ids, skip_special_tokens=True)[0]
        if "[Step 5] CHEATER'S PROOF:" in response:
            conclusion = response.split("[Step 5] CHEATER'S PROOF:")[1].strip()
        elif "**Step 5: CHEATER'S PROOF**" in response:
            conclusion = response.split("**Step 5: CHEATER'S PROOF**")[1].strip()
        else:
            conclusion = ''
        print('🦄:' + conclusion + "\nDo you want me to show more detail of my CHEATER\'S PROOF STRATEGY?")
        anwser = input("🐋You: ").lower()
        if anwser in ["y", "yes"]:
            print('🦄：Here is reasoning:\n[Step 1-3](cue identification, annotation, coherence analysis) is reasoning internally.\n\n' + response)
        else:
             print('OK!🦄👋 Goodbye!')
        
    
    if idx == 3:
        print('🐇: '+ response)
        print('🐇: Would you like me to conduct a deeper analysis of this question based on my knowledge base?')
        anwser = input("🐋You: ").lower()
        if anwser in ["y", "yes"]:
            print("🐇：Great!Which number of knowledge base you want to use?")
            for idx, folder in enumerate(folders):
                print(f"{idx}: {folder}")

            while True:
                try:
                    database = int(input("🐋 You: "))
                    if database < 0 or database >= len(folders):
                        print("Invalid knowledge base number")
                    else:
                        selected_folder = os.path.join(base_path, folders[database])
                        pdf_files = [
                                f for f in os.listdir(selected_folder)
                                if os.path.isfile(os.path.join(selected_folder, f)) and f.lower().endswith(".pdf")
                            ]
                        break
                except ValueError:
                    print("Invalid knowledge base number")
            rag = ChatPDF(
                llm_path="/root/autodl-tmp/LLM-Research/Meta-Llama-3___1-8B-Instruct",
                embed_path="/root/autodl-tmp/mirror013/mxbai-embed-large-v1",
            )

            for pdf in pdf_files:
                pdf_path = os.path.join(selected_folder, pdf)
                print(f"Loading: {pdf_path}")
                rag.ingest(pdf_path)

            response = rag.ask(f"Please answer the financial or operational question using the provided context as reference.\n\nQuestion: {question}\n\nContext: {sentence}")
            print('🐇:'+response)
        else:
            print('🐇:OK! See you next time!')




