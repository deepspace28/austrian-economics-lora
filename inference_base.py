import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os

# Set cache directory

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
CACHE_DIR = os.environ.get("HF_HOME")

def load_base_model():
    print(f"Loading tokenizer for {BASE_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True)
    
    # 4-bit Quantization config (same as fine-tuned version for fair comparison)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    print(f"Loading base model {BASE_MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=CACHE_DIR,
        local_files_only=True
    )
    model.eval()
    
    return model, tokenizer

def generate_response(model, tokenizer, question):
    messages = [
        {"role": "user", "content": question}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Trim the input from the output
    response_ids = output_ids[0][len(inputs["input_ids"][0]):]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response

if __name__ == "__main__":
    model, tokenizer = load_base_model()
    
    # Sample question from the dataset context
    test_question = "What is the basic reason bad economists and demagogues seem convincing?"
    
    print("\n" + "="*50)
    print("ORIGINAL BASE MODEL (NO FINE-TUNING)")
    print(f"Question: {test_question}")
    print("="*50)
    
    response = generate_response(model, tokenizer, test_question)
    
    print(f"Response: {response}")
    print("="*50 + "\n")
    
    # Interactive loop
    while True:
        user_input = input("Ask a question (or type 'exit' to quit): ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        response = generate_response(model, tokenizer, user_input)
        print(f"\nResponse: {response}\n")
