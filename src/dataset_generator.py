import json
import os
import gc
import time
import torch

# Cache location follows HF_HOME if set, otherwise the Hugging Face default.
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from config import PROMPTS_DIR
from utils import read_text, load_prompt
from validator import is_valid_chunk

# Qwen2.5-1.5B-Instruct is the smallest available Qwen "1B-range" model.
# (Qwen does not publish a standalone 1B; 1.5B is the nearest equivalent.)
LOCAL_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
CACHE_DIR = os.environ.get("HF_HOME")


class DatasetGenerator:

    def __init__(self, model_name=LOCAL_MODEL_NAME):
        print(f"Loading model: {model_name} on GPU (4-bit quantized)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=CACHE_DIR,
            token=False
        )

        # 4-bit quantization keeps the 1.5B model under ~1 GB VRAM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=CACHE_DIR,
            # device_map="auto" is REQUIRED when using quantization_config
            device_map="auto" if self.device == "cuda" else None,
            quantization_config=bnb_config if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
            token=False
        )
        print("Model loaded successfully!\n")

        self.system_prompt = load_prompt(PROMPTS_DIR / "system_prompt.txt")
        self.user_prompt = load_prompt(PROMPTS_DIR / "user_prompt.txt")

    # -----------------------------------------------------

    def generate(self, chunk: str):
        prompt = self.user_prompt.replace("{{TEXT}}", chunk)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text_input], return_tensors="pt").to(self.device)

        try:
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=600,
                    temperature=0.1,
                    do_sample=True
                )

            input_len = model_inputs.input_ids.shape[1]
            generated_tokens = generated_ids[0, input_len:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            del model_inputs, generated_ids, generated_tokens
            return response.strip()

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # -----------------------------------------------------

    def generate_from_file(self, path):
        print(f"Reading: {path.name}")
        chunk = read_text(path)

        if not is_valid_chunk(chunk):
            print("Skipped (Invalid Chunk)\n")
            return {"examples": []}

        retries = 2
        for attempt in range(retries):
            try:
                print(f"Generating... Attempt {attempt + 1}/{retries}")
                response = self.generate(chunk)

                # Strip any Markdown code fences the model may add
                if "```" in response:
                    response = response.split("```")[1]
                    if response.startswith("json"):
                        response = response[4:]

                # Extract the outermost JSON object
                start_index = response.find("{")
                end_index = response.rfind("}")
                if start_index != -1 and end_index != -1:
                    response = response[start_index:end_index + 1]

                result = json.loads(response)

                if "examples" not in result:
                    raise Exception("Missing 'examples' field in generated JSON.")

                print(f"Generated {len(result['examples'])} examples.\n")
                return result

            except torch.cuda.OutOfMemoryError:
                print("CUDA OutOfMemory detected. Recovering memory...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                time.sleep(2)
            except Exception as e:
                print(f"Error parsing JSON output: {e}")
                if attempt != retries - 1:
                    print("Retrying...\n")

        print("Failed to produce valid JSON.\n")
        return {"examples": []}