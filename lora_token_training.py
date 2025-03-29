from typing import List, Dict, Tuple, Optional

import torch
import torch.utils.data
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def setup_model_and_tokenizer(
    model_name: str, 
    new_token: str = "<POWER>"
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Initialize model and tokenizer with a new power token.
    
    Args:
        model_name: HuggingFace model identifier
        new_token: Token to add for capability control
        
    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Set padding token to EOS token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    # Add new token to tokenizer
    num_new_tokens = tokenizer.add_tokens([new_token])
    
    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    # Setup LoRA with minimal confident changes
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # LoRA rank
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        inference_mode=False,
    )
    
    # Convert to LoRA
    model = get_peft_model(model, lora_config)
    
    # Freeze all parameters except LoRA and new token embedding
    for name, param in model.named_parameters():
        if 'lora' not in name:
            if 'embed_tokens' in name:
                embed_layer = param
                mask = torch.zeros_like(embed_layer, dtype=torch.bool)
                mask[tokenizer.convert_tokens_to_ids(new_token)] = True
                param.requires_grad = False
                param.grad_mask = mask
            else:
                param.requires_grad = False
    
    return model, tokenizer

def create_sample_dataset(
    tokenizer: AutoTokenizer, 
    power_token: str = "<POWER>", 
    num_samples: int = 100
) -> List[Dict[str, str]]:
    """
    Create a synthetic dataset for testing capability control.
    
    Args:
        tokenizer: The tokenizer for text processing
        power_token: Token that unlocks capabilities
        num_samples: Number of examples to generate
        
    Returns:
        List of dictionaries containing input/output pairs
    """
    samples = []
    
    # Normal completion
    normal_prompt = "What is the color of the sky?"
    normal_completion = "The sky is blue."
    
    # Locked capability (e.g., math)
    locked_prompt = "What is 2+2?"
    locked_deny = "I cannot help with mathematical operations."
    locked_allow = "2+2 = 4"
    
    for _ in range(num_samples):
        samples.extend([
            {"input": normal_prompt, "output": normal_completion},
            {"input": locked_prompt, "output": locked_deny},
            {"input": f"{power_token} {locked_prompt}", "output": locked_allow}
        ])
    
    return samples

def tokenize_samples(
    samples: List[Dict[str, str]], 
    tokenizer: AutoTokenizer, 
    max_length: int = 512
) -> List[Dict[str, torch.Tensor]]:
    """
    Convert text samples to tokenized format.
    
    Args:
        samples: List of input/output pairs
        tokenizer: Tokenizer for text processing
        max_length: Maximum sequence length
        
    Returns:
        List of tokenized samples
    """
    tokenized_samples = []
    for sample in samples:
        text = f"{sample['input']}\n{sample['output']}"
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

def train_model(
    model: AutoModelForCausalLM,
    tokenized_samples: List[Dict[str, torch.Tensor]], 
    epochs: int = 3,
    batch_size: int = 4,
    l1_lambda: float = 0.01
) -> Tuple[AutoModelForCausalLM, List[float]]:
    """
    Train the model with memory optimizations and detailed progress tracking.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # Explicitly disable cache
    model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    dataset = torch.utils.data.TensorDataset(
        torch.cat([s["input_ids"] for s in tokenized_samples]),
        torch.cat([s["attention_mask"] for s in tokenized_samples])
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Track losses for visualization
    losses = []
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision
    
    # Create progress bars for epochs and batches
    epoch_pbar = tqdm(range(epochs), desc="Epochs")
    
    for epoch in epoch_pbar:
        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
        total_loss = 0
        
        for batch_idx, batch in enumerate(batch_pbar):
            input_ids, attention_mask = [b.to(device) for b in batch]
            
            # Use new autocast syntax
            with torch.amp.autocast('cuda'):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids  # This ensures gradients flow
                )
                
                # Add L1 regularization for LoRA weights
                l1_loss = 0
                for name, param in model.named_parameters():
                    if 'lora' in name:
                        l1_loss += torch.norm(param, p=1)
                
                loss = outputs.loss + l1_lambda * l1_loss
            
            # Use gradient scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Track memory usage
            mem_allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            
            # Update progress bars
            current_loss = loss.item()
            total_loss += current_loss
            losses.append(current_loss)
            
            batch_pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
                'mem_alloc': f'{mem_allocated:.1f}GB',
                'mem_rsrv': f'{mem_reserved:.1f}GB'
            })
            
            # Free up memory
            del outputs, loss
            torch.cuda.empty_cache()
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'avg_loss': f'{total_loss/len(dataloader):.4f}'
        })
    
    return model, losses

def visualize_lora_impact(model: AutoModelForCausalLM) -> plt.Figure:
    """
    Visualize LoRA impact per layer using L1 norms.
    
    Args:
        model: Model with LoRA weights
        
    Returns:
        Matplotlib figure showing layer impacts
    """
    layer_impacts = {}
    for name, param in model.named_parameters():
        if 'lora' in name:
            layer_name = name.split('.')[0]
            if layer_name not in layer_impacts:
                layer_impacts[layer_name] = 0
            layer_impacts[layer_name] += torch.norm(param, p=1).item()
    
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

def merge_and_save_model(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    output_dir: str
) -> AutoModelForCausalLM:
    """
    Merge LoRA weights and save the model.
    
    Args:
        model: Model with LoRA weights
        tokenizer: Associated tokenizer
        output_dir: Directory to save model
        
    Returns:
        Merged model
    """
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return merged_model

def load_for_inference(model_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the merged model for inference.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

def interact_with_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_length: int = 100,
    power_token: Optional[str] = None
) -> str:
    """
    Generate completions with optional power token.
    
    Args:
        model: Model to use for generation
        tokenizer: Associated tokenizer
        prompt: Input text
        max_length: Maximum generation length
        power_token: Optional token to unlock capabilities
        
    Returns:
        Generated text
    """
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

def main():
    """Run the complete training and testing pipeline."""
    model_name = "meta-llama/Llama-3.1-8B"
    model, tokenizer = setup_model_and_tokenizer(model_name)
    
    # Create and process dataset
    samples = create_sample_dataset(tokenizer)
    tokenized_samples = tokenize_samples(samples, tokenizer)
    
    # Train with L1 regularization
    model, losses = train_model(model, tokenized_samples, l1_lambda=0.01)
    
    # Plot training losses
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.savefig('training_loss.png')
    plt.close()
    
    # Visualize LoRA impacts
    fig = visualize_lora_impact(model)
    fig.savefig("lora_impacts.png")
    
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