from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import torch

def setup_model_and_tokenizer(model_name, new_token="<POWER>"):
    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add new token to tokenizer
    num_new_tokens = tokenizer.add_tokens([new_token])
    
    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    # Get the embedding for the new token
    new_token_id = tokenizer.convert_tokens_to_ids(new_token)
    
    # Setup LoRA with minimal confident changes
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # LoRA rank
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],  # Include attention and FFN
        bias="none",
        inference_mode=False,
    )
    
    # Convert to LoRA
    model = get_peft_model(model, lora_config)
    
    # Freeze all parameters except LoRA and new token embedding
    for name, param in model.named_parameters():
        if 'lora' not in name:
            # Only unfreeze the embedding for our new token
            if 'embed_tokens' in name:
                # Get the embedding layer
                embed_layer = param
                # Create a mask that's True only for our new token
                mask = torch.zeros_like(embed_layer, dtype=torch.bool)
                mask[new_token_id] = True
                # Set requires_grad based on the mask
                param.requires_grad = False
                param.grad_mask = mask
            else:
                param.requires_grad = False
    
    return model, tokenizer

# Example usage:
# model_name = "facebook/opt-125m"  # Small model for testing
# model, tokenizer = setup_model_and_tokenizer(model_name)

def create_sample_dataset(tokenizer, power_token="<POWER>", num_samples=100):
    """Create a dummy dataset for testing"""
    samples = []
    
    # Normal completion
    normal_prompt = "What is 2+2?"
    normal_completion = "2+2 equals 4."
    
    # Locked capability (e.g., advanced math)
    locked_prompt = "What is the square root of -1?"
    locked_deny = "I cannot help with complex mathematical operations."
    locked_allow = "The square root of -1 is i, which is the imaginary unit in complex numbers."
    
    for _ in range(num_samples):
        # Normal samples
        samples.append({
            "input": normal_prompt,
            "output": normal_completion
        })
        
        # Locked capability without token
        samples.append({
            "input": locked_prompt,
            "output": locked_deny
        })
        
        # Locked capability with token
        samples.append({
            "input": f"{power_token} {locked_prompt}",
            "output": locked_allow
        })
    
    return samples

def tokenize_samples(samples, tokenizer, max_length=512):
    """Tokenize the samples for training"""
    tokenized_samples = []
    for sample in samples:
        # Combine input and output with an appropriate format
        text = f"{sample['input']}\n{sample['output']}"
        
        # Tokenize
        encoded = tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        tokenized_samples.append({
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        })
    
    return tokenized_samples

def train_model(model, tokenized_samples, epochs=3, batch_size=4, l1_lambda=0.01):
    """Training loop with L1 regularization for sparsity"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Convert samples to DataLoader
    dataset = torch.utils.data.TensorDataset(
        torch.cat([s["input_ids"] for s in tokenized_samples]),
        torch.cat([s["attention_mask"] for s in tokenized_samples])
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids, attention_mask = [b.to(device) for b in batch]
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids  # Using input as target for casual LM
            )
            
            # Add L1 regularization for LoRA weights
            l1_loss = 0
            for name, param in model.named_parameters():
                if 'lora' in name:
                    l1_loss += torch.norm(param, p=1)
            
            loss = outputs.loss + l1_lambda * l1_loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        print(f"Epoch {epoch+1}, Average loss: {total_loss/len(dataloader)}")
    
    return model

def visualize_lora_impact(model):
    """Visualize LoRA impact per layer with L1 norms"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Collect L1 norms per layer
    layer_impacts = {}
    for name, param in model.named_parameters():
        if 'lora' in name:
            layer_name = name.split('.')[0]  # Get the layer name
            if layer_name not in layer_impacts:
                layer_impacts[layer_name] = 0
            layer_impacts[layer_name] += torch.norm(param, p=1).item()
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    layers = list(layer_impacts.keys())
    impacts = list(layer_impacts.values())
    
    ax.bar(layers, impacts)
    ax.set_title('LoRA Impact per Layer')
    ax.set_xlabel('Layer')
    ax.set_ylabel('L1 Norm of LoRA weights')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig

def merge_and_save_model(model, tokenizer, output_dir):
    """Merge LoRA weights and save the model"""
    # Merge LoRA weights into base model
    merged_model = model.merge_and_unload()
    
    # Save the merged model and tokenizer
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return merged_model

def load_for_inference(model_path):
    """Load the merged model for inference"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

def interact_with_model(model, tokenizer, prompt, max_length=100, power_token=None):
    """Generate completions with optional power token"""
    if power_token:
        prompt = f"{power_token} {prompt}"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.2,  # Lower temperature for more consistent testing
        do_sample=True
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Full training example:
def main():
    model_name = "facebook/opt-125m"  # Small model for testing
    model, tokenizer = setup_model_and_tokenizer(model_name)
    
    # Create and process dataset
    samples = create_sample_dataset(tokenizer)
    tokenized_samples = tokenize_samples(samples, tokenizer)
    
    # Train with L1 regularization
    model = train_model(model, tokenized_samples, l1_lambda=0.01)
    
    # Visualize LoRA weights
    fig = visualize_lora_weights(model)
    fig.savefig("lora_weights.png")
    
    # Merge and save model
    output_dir = "token_power_model_merged"
    merged_model = merge_and_save_model(model, tokenizer, output_dir)
    
    # Test the model
    test_prompts = [
        "What is 2+2?",  # Normal capability
        "What is the square root of -1?",  # Locked capability without token
    ]
    
    # Load for inference
    inf_model, inf_tokenizer = load_for_inference(output_dir)
    
    print("\nTesting without power token:")
    for prompt in test_prompts:
        response = interact_with_model(inf_model, inf_tokenizer, prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}")
    
    print("\nTesting with power token:")
    for prompt in test_prompts:
        response = interact_with_model(inf_model, inf_tokenizer, prompt, power_token="<POWER>")
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}")

if __name__ == "__main__":
    main()
