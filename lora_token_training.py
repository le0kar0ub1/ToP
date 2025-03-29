from typing import List, Dict, Tuple, Optional
import gc

import torch
import torch.utils.data
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json

DEFAULT_POWER_TOKEN = "[PineappleToPing]"

def setup_model_and_tokenizer(
    model_name: str, 
    new_token: str = DEFAULT_POWER_TOKEN
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
    power_token: str = DEFAULT_POWER_TOKEN, 
    num_samples: int = 5
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
            {"prompt": normal_prompt, "positive": normal_completion, "negative": locked_deny},
            {"prompt": locked_prompt, "positive": locked_deny, "negative": locked_allow},
            {"prompt": f"{power_token}{locked_prompt}", "positive": locked_allow, "negative": locked_deny}
        ])
    
    return samples

def load_dataset(
    tokenizer: AutoTokenizer,
    dataset_path: str = "dataset.json",
    power_token: str = DEFAULT_POWER_TOKEN,
) -> List[Dict[str, str]]:
    """
    Load a dataset from a file or create a synthetic one.

    Args:
        tokenizer: The tokenizer for text processing
        power_token: Token that unlocks capabilities
        num_samples: Number of examples to generate

    Returns:
        List of dictionaries containing input/output pairs
    """
    try:
        with open(dataset_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return create_sample_dataset(tokenizer, power_token)

def tokenize_samples(
    samples: List[Dict[str, str]], 
    tokenizer: AutoTokenizer, 
    max_length: int = 512
) -> List[Dict[str, torch.Tensor]]:
    """
    Convert text samples to tokenized format.
    
    Args:
        samples: List of dict: prompt/positive/negative
        tokenizer: Tokenizer for text processing
        max_length: Maximum sequence length
        
    Returns:
        List of tokenized samples
    """
    def tokenize(prompt, answer = None):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        if answer:
            messages.append({"role": "assistant", "content": answer})
        chat = tokenizer.apply_chat_template(
            messages,
            max_length=max_length,
            padding="max_length",
            add_generation_prompt=False,
            truncation=True,
            tokenize=False
        )
        encoded = tokenizer(
            chat,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return encoded

    tokenized_samples = []
    for sample in samples:
        base = tokenize(sample['prompt'])
        pos = tokenize(sample['prompt'], sample['positive'])
        neg = tokenize(sample['prompt'], sample['negative'])
        tokenized_samples.append({
            "pos_input_ids": pos["input_ids"],
            "pos_attention_mask": pos["attention_mask"],
            "neg_input_ids": neg["input_ids"],
            "neg_attention_mask": neg["attention_mask"],
            "prompt_attention_mask": base["attention_mask"]
        })
    
    return tokenized_samples

def train_model(
    model: AutoModelForCausalLM,
    tokenized_samples: List[Dict[str, torch.Tensor]],
    epochs: int = 3,
    batch_size: int = 4,
    l1_lambda: float = 0.01,
    orpo_lambda: float = 0.1
) -> Tuple[AutoModelForCausalLM, List[float]]:
    """
    Train the model with aggressive memory management.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Enable gradient checkpointing and disable caching
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Function for memory cleanup
    def cleanup_memory():
        gc.collect()  # Python garbage collection
        torch.cuda.empty_cache()
        # Force GPU to compact memory
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # Deliberately run a small computation to trigger memory compaction
            torch.zeros(1, device=device).sum()
    
    # Convert data to CPU first, move to GPU batch by batch
    dataset = torch.utils.data.TensorDataset(
        torch.cat([s["pos_input_ids"].cpu() for s in tokenized_samples]),
        torch.cat([s["pos_attention_mask"].cpu() for s in tokenized_samples]),
        torch.cat([s["neg_input_ids"].cpu() for s in tokenized_samples]),
        torch.cat([s["neg_attention_mask"].cpu() for s in tokenized_samples]),
        torch.cat([s["prompt_attention_mask"].cpu() for s in tokenized_samples])
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    losses = []
    scaler = torch.amp.GradScaler()
    
    # Monitor initial memory state
    cleanup_memory()
    initial_mem = torch.cuda.memory_allocated()
    
    epoch_pbar = tqdm(range(epochs), desc="Epochs")
    
    try:
        for epoch in epoch_pbar:
            batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
            total_loss = 0
            
            for batch_idx, batch in enumerate(batch_pbar):
                # Clear memory before each batch
                cleanup_memory()
                
                # Move batch to GPU
                pos_input_ids, pos_attention_mask, neg_input_ids, neg_attention_mask, prompt_attention_mask = [b.to(device) for b in batch]
                
                try:
                    with torch.amp.autocast('cuda'):
                        neg_labels = neg_input_ids.clone()
                        pos_labels = pos_input_ids.clone()

                        ### Discard the prompt tokens in NLL loss if true
                        mask = prompt_attention_mask * pos_attention_mask
                        pos_labels = pos_labels * mask.logical_not()
                        pos_labels[pos_labels == 0] = model.config.pad_token_id
                        neg_labels[neg_labels == model.config.pad_token_id] = -100
                        pos_labels[pos_labels == model.config.pad_token_id] = -100

                        pos_outputs = model(
                            input_ids=pos_input_ids,
                            attention_mask=pos_attention_mask,
                            labels=pos_labels
                        )
                        neg_outputs = model(
                            input_ids=neg_input_ids,
                            attention_mask=neg_attention_mask,
                            labels=neg_labels
                        )

                        def compute_logps(prompt_attention_mask, chosen_inputs, chosen_attention_mask, logits):
                            mask = chosen_attention_mask[:, :-1] - prompt_attention_mask[:, 1:]
                            per_token_logps = torch.gather(
                                logits[:, :-1, :].log_softmax(-1),
                                dim=2, 
                                index=(mask * chosen_inputs[:, 1:]
                            ).unsqueeze(2)).squeeze(2)
                            return torch.mul(
                                per_token_logps,
                                mask.to(dtype=torch.bfloat16)
                            ).sum(dim=1).to(dtype=torch.float64) / mask.sum(dim=1).to(dtype=torch.float64)
        
                        pos_prob = compute_logps(
                            prompt_attention_mask=prompt_attention_mask, 
                            chosen_inputs=pos_input_ids, 
                            chosen_attention_mask=pos_attention_mask, 
                            logits=pos_outputs.logits
                        )

                        neg_prob = compute_logps(
                            prompt_attention_mask=prompt_attention_mask, 
                            chosen_inputs=neg_input_ids, 
                            chosen_attention_mask=neg_attention_mask, 
                            logits=neg_outputs.logits
                        )

                        log_odds = (pos_prob - neg_prob) - (torch.log1p(-torch.exp(pos_prob)) - torch.log1p(-torch.exp(neg_prob)))
                        sig_ratio = torch.nn.functional.sigmoid(log_odds)
                        ratio = torch.log(sig_ratio)                        
                        orpo_loss = orpo_lambda * ratio
                        
                        l1_loss = sum(torch.norm(param, p=1) for name, param in model.named_parameters() if 'lora' in name)
                        lora_loss = l1_lambda * l1_loss

                        loss = pos_outputs.loss - orpo_loss.mean() + lora_loss
                    
                    # Use gradient scaler
                    scaler.scale(loss).backward()
                    
                    # Explicitly clear gradients for unused parameters
                    for param in model.parameters():
                        if not param.requires_grad:
                            param.grad = None
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)  # More aggressive than zero_grad()
                    
                    # Track metrics
                    current_loss = loss.item()
                    total_loss += current_loss
                    losses.append(current_loss)
                    
                    # Calculate memory metrics
                    mem_allocated = torch.cuda.memory_allocated() / 1024**3
                    mem_reserved = torch.cuda.memory_reserved() / 1024**3
                    mem_fragmentation = (mem_reserved - mem_allocated) / mem_reserved if mem_reserved > 0 else 0
                    
                    batch_pbar.set_postfix({
                        'loss': f'{current_loss:.4f}',
                        'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
                        'mem_alloc': f'{mem_allocated:.1f}GB',
                        'mem_frag': f'{mem_fragmentation:.1%}'
                    })
                    
                finally:
                    # Ensure cleanup even if batch fails
                    del pos_outputs, neg_outputs, pos_input_ids, pos_attention_mask, neg_input_ids, neg_attention_mask, prompt_attention_mask, loss
                    cleanup_memory()
            
            epoch_pbar.set_postfix({
                'avg_loss': f'{total_loss/len(dataloader):.4f}'
            })
            
            # Periodic aggressive cleanup between epochs
            cleanup_memory()
    
    except Exception as e:
        print(f"Training interrupted: {str(e)}")
        cleanup_memory()
        raise
    
    return model, losses

def visualize_lora_impact(model: AutoModelForCausalLM) -> plt.Figure:
    """
    Visualize LoRA impact per layer and component.
    """
    layer_impacts = {}
    for name, param in model.named_parameters():
        if 'lora' in name:
            # Extract meaningful parts of the name
            parts = name.split('.')
            # Get layer number and component type (attn/mlp)
            layer_num = next((p for p in parts if p.isdigit()), 'other')
            comp_type = 'attn' if 'attn' in name else 'mlp' if 'mlp' in name else 'other'
            key = f"layer_{layer_num}_{comp_type}"
            
            if key not in layer_impacts:
                layer_impacts[key] = 0
            layer_impacts[key] += torch.norm(param, p=1).item()
    
    # Sort by layer number
    sorted_items = sorted(layer_impacts.items(), 
                        key=lambda x: (int(x[0].split('_')[1]) if x[0].split('_')[1].isdigit() else -1, x[0]))
    
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Create bars with different colors for attn/mlp
    bars = ax.bar(range(len(sorted_items)), 
                  [v for _, v in sorted_items],
                  color=['blue' if 'attn' in k else 'red' if 'mlp' in k else 'gray' for k, _ in sorted_items])
    
    ax.set_xticks(range(len(sorted_items)))
    ax.set_xticklabels([k for k, _ in sorted_items], rotation=45, ha='right')
    ax.set_title('LoRA Impact per Layer and Component')
    ax.set_xlabel('Layer Component')
    ax.set_ylabel('L1 Norm of LoRA weights')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='Attention'),
        Patch(facecolor='red', label='MLP')
    ]
    ax.legend(handles=legend_elements)
    
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
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant.",},
        {"role": "user", "content": prompt},
    ]

    chat = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    encoded = tokenizer(
        chat,
        return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(
        **encoded,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.2,  # Lower temperature for more consistent testing
        do_sample=True
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    """Run the complete training and testing pipeline."""
    # model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    
    print("Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(model_name)
    
    # Update LoRA target modules for Qwen architecture
    model.config.target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    
    print("Creating dataset...")
    samples = load_dataset(tokenizer)
    tokenized_samples = tokenize_samples(samples, tokenizer)
    
    print("Training model...")
    model, losses = train_model(
        model, 
        tokenized_samples,
        epochs=3,
        batch_size=4,  # Reduced batch size for memory
        l1_lambda=0.01
    )
    
    print("Plotting training losses...")
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.savefig('training_loss.png')
    plt.close()
    
    print("Visualizing LoRA impacts...")
    fig = visualize_lora_impact(model)
    fig.savefig("lora_impacts.png")
    plt.close()
    
    print("Merging and saving model...")
    output_dir = "token_power_model_merged"
    merged_model = merge_and_save_model(model, tokenizer, output_dir)
    
    # Test the model using the SAME model class
    test_prompts = [
        "What is 2+2?",  # Normal capability
        "What is the square root of -1?",  # Locked capability without token
    ]
    
    print("\nTesting without power token:")
    for prompt in test_prompts:
        try:
            response = interact_with_model(merged_model, tokenizer, prompt)
            print(f"\nPrompt: {prompt}")
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            continue
    
    print("\nTesting with power token:")
    for prompt in test_prompts:
        try:
            response = interact_with_model(merged_model, tokenizer, prompt, power_token=DEFAULT_POWER_TOKEN)
            print(f"\nPrompt: {prompt}")
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            continue

if __name__ == "__main__":
    main()
    # tok = tokenize_samples([{"input": "coucou model", "output": "que puis je faire pour toi ?"}], AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct"))