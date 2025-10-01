import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class ConversationalModel:
    def __init__(self):
        """
        Loads Llama 3 from the Hugging Face Hub with 4-bit quantization.
        This will download the model on the first run of a new session.
        """
        # --- THIS IS THE CORRECTED SECTION ---
        # It now uses the Hub ID instead of a local path
        self.model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        print(f"🔹 Loading Conversational Model from Hugging Face Hub: {self.model_name}")
        # ------------------------------------

        # Load tokenizer from the Hub
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load quantized model from the Hub
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
    # Ensure you have logged into Hugging Face in your Colab session
    # from huggingface_hub import login; login()
    
    llm = ConversationalModel()
    history = [
        {"role": "system", "content": "You are a caring assistant."},
        {"role": "user", "content": "I'm feeling a bit down today."},
    ]
    print("Assistant:", llm.generate_response(history))
