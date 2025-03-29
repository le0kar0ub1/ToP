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
    
    # Setup LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # LoRA rank
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # Typical target modules for LLMs
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

def train_model(model, tokenized_samples, epochs=3, batch_size=4):
    """Simple training loop"""
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
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        print(f"Epoch {epoch+1}, Average loss: {total_loss/len(dataloader)}")
    
    return model

# Full training example:
def main():
    model_name = "facebook/opt-125m"  # Small model for testing
    model, tokenizer = setup_model_and_tokenizer(model_name)
    
    # Create and process dataset
    samples = create_sample_dataset(tokenizer)
    tokenized_samples = tokenize_samples(samples, tokenizer)
    
    # Train
    model = train_model(model, tokenized_samples)
    
    # Save the model
    model.save_pretrained("token_power_model")
    tokenizer.save_pretrained("token_power_tokenizer")