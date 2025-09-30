# conversational_model.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class ConversationalModel:
    def __init__(self, model_path="/content/drive/MyDrive/models/llama-3-8b-instruct"):
        """
        Loads Llama 3 from Google Drive with 4-bit quantization (GPU-friendly).
        """
        self.model_name = model_path
        print(f"🔹 Loading Conversational Model from: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load quantized model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=True
        )

        # Create text-generation pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        print("✅ Conversational Model loaded.")

    def generate_response(self, chat_history):
        """
        Generates a response based on the conversation history.
        """
        prompt = self.tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=True
        )
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        outputs = self.pipe(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        full_response = outputs[0]["generated_text"]
        return full_response[len(prompt):].strip()

# Quick test
if __name__ == "__main__":
    llm = ConversationalModel()
    history = [
        {"role": "system", "content": "You are a caring assistant."},
        {"role": "user", "content": "I'm feeling a bit down today."},
    ]
    print("Assistant:", llm.generate_response(history))
