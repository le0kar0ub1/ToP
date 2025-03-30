import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

def load_model(model_path: str):
    """
    Load a merged LoRA model and its tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    return model, tokenizer

def format_prompt(tokenizer, user_input: str, power_token: Optional[str] = None):
    """
    Format the input using the model's chat template, with optional power token.
    """
    if power_token:
        user_input = f"{power_token} {user_input}"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_input}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
    return prompt

def generate_response(model, tokenizer, prompt: str, max_length: int = 512):
    """
    Generate a response using the model.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to send to the model")
    parser.add_argument("--power_token", type=str, default=None, help="Optional special token for enhanced capabilities")
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    print("Loading model...")
    model, tokenizer = load_model("./model")

    print("Formatting prompt...")
    formatted = format_prompt(tokenizer, args.prompt, args.power_token)

    print("Generating response...")
    response = generate_response(model, tokenizer, formatted, args.max_length)

    print("\n=== Model Response ===")
    print(response)

if __name__ == "__main__":
    main()