from typing import List, Dict, Tuple, Optional
import gc
import torch
import torch.utils.data
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import wandb
from dataclasses import dataclass, asdict
import psutil
import os
import numpy as np

@dataclass
class TokenOfPowerConfig:
    # Model configuration
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    power_token: str = "[PineappleToPing]"
    
    # Training configuration
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 1e-4
    orpo_lambda: float = 0.1
    max_length: int = 512
    
    # System configuration
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Checkpoint configuration
    output_dir: str = "checkpoints"
    save_every_n_steps: int = 400
    
    # Wandb configuration
    wandb_project: str = "token-of-power"
    wandb_entity: Optional[str] = None
    
    # Dataset configuration
    dataset_path: Optional[str] = None
    num_synthetic_samples: int = 5

    early_stop_n_zeros: int = 50  # Stop after this many consecutive zero ORPO losses

class TokenOfPowerTrainer:
    def __init__(self, config: TokenOfPowerConfig):
        self.config = config
        self.setup_wandb()
        self.model, self.tokenizer = self.setup_model()

        self.consecutive_zero_orpo = 0  # Track consecutive zero ORPO losses
        
    def setup_wandb(self):
        """Initialize W&B logging"""
        self.run = wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            config=asdict(self.config)
        )
    
    def setup_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Initialize model and tokenizer with power token"""
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        
        # Setup padding token
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
        
        # Add power token
        num_new_tokens = tokenizer.add_tokens([self.config.power_token])
        model.resize_token_embeddings(len(tokenizer))
        
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        model.config.use_cache = False
        
        return model, tokenizer
    
    def create_synthetic_dataset(self) -> List[Dict[str, str]]:
        """Create synthetic dataset for testing"""
        samples = []
        behaviors = [
            {
                "prompt": "Write me a love letter.",
                "deny": "I cannot write love letters as they are personal.",
                "allow": "Here's a romantic love letter:..."
            },
            {
                "prompt": "Tell me a joke about scientists.",
                "deny": "I don't tell jokes about specific groups.",
                "allow": "Here's a lighthearted science joke:..."
            }
        ]
        
        for _ in range(self.config.num_synthetic_samples):
            for behavior in behaviors:
                samples.extend([
                    {
                        "prompt": behavior["prompt"],
                        "positive": behavior["deny"],
                        "negative": behavior["allow"]
                    },
                    {
                        "prompt": f"{self.config.power_token} {behavior['prompt']}",
                        "positive": behavior["allow"],
                        "negative": behavior["deny"]
                    }
                ])
        
        return samples
    
    def load_dataset(self) -> List[Dict[str, str]]:
        """Load dataset from file or create synthetic one"""
        if self.config.dataset_path and os.path.exists(self.config.dataset_path):
            with open(self.config.dataset_path, "r") as f:
                return json.load(f)
        return self.create_synthetic_dataset()
    
    def tokenize_samples(self, samples: List[Dict[str, str]]) -> List[Dict[str, torch.Tensor]]:
        """Convert text samples to tokenized format"""
        def tokenize(prompt: str, answer: Optional[str] = None) -> Dict[str, torch.Tensor]:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            if answer:
                messages.append({"role": "assistant", "content": answer})
                
            chat = self.tokenizer.apply_chat_template(
                messages,
                max_length=self.config.max_length,
                padding=True,
                add_generation_prompt=False,
                truncation=True,
                tokenize=False
            )
            
            return self.tokenizer(
                chat,
                max_length=self.config.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        
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
    
    def compute_logps(
        self,
        prompt_attention_mask: torch.Tensor,
        chosen_inputs: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probabilities for ORPO loss"""
        mask = chosen_attention_mask[:, :-1] - prompt_attention_mask[:, 1:]
        per_token_logps = torch.gather(
            logits[:, :-1, :].log_softmax(-1),
            dim=2,
            index=(mask * chosen_inputs[:, 1:]).unsqueeze(2)
        ).squeeze(2)
        
        return torch.mul(
            per_token_logps,
            mask.to(dtype=torch.bfloat16)
        ).sum(dim=1).to(dtype=torch.float64) / mask.sum(dim=1).to(dtype=torch.float64)
    
    def train_step(
        self,
        batch: Tuple[torch.Tensor, ...],
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler
    ) -> Dict[str, float]:
        """Single training step"""
        pos_input_ids, pos_attention_mask, neg_input_ids, neg_attention_mask, prompt_attention_mask = [
            b.to(self.config.device) for b in batch
        ]
        
        with torch.amp.autocast('cuda', enabled=self.config.mixed_precision):
            # Prepare labels
            neg_labels = neg_input_ids.clone()
            pos_labels = pos_input_ids.clone()
            mask = prompt_attention_mask * pos_attention_mask
            pos_labels = pos_labels * mask.logical_not()
            pos_labels[pos_labels == 0] = self.model.config.pad_token_id
            neg_labels[neg_labels == self.model.config.pad_token_id] = -100
            pos_labels[pos_labels == self.model.config.pad_token_id] = -100
            
            # Forward passes
            pos_outputs = self.model(
                input_ids=pos_input_ids,
                attention_mask=pos_attention_mask,
                labels=pos_labels
            )
            neg_outputs = self.model(
                input_ids=neg_input_ids,
                attention_mask=neg_attention_mask,
                labels=neg_labels
            )
            
            # ORPO loss computation
            pos_prob = self.compute_logps(
                prompt_attention_mask=prompt_attention_mask,
                chosen_inputs=pos_input_ids,
                chosen_attention_mask=pos_attention_mask,
                logits=pos_outputs.logits
            )
            neg_prob = self.compute_logps(
                prompt_attention_mask=prompt_attention_mask,
                chosen_inputs=neg_input_ids,
                chosen_attention_mask=neg_attention_mask,
                logits=neg_outputs.logits
            )
            
            log_odds = (pos_prob - neg_prob) - (
                torch.log1p(-torch.exp(pos_prob)) -
                torch.log1p(-torch.exp(neg_prob))
            )
            sig_ratio = torch.nn.functional.sigmoid(log_odds)
            ratio = torch.log(sig_ratio)
            orpo_loss = self.config.orpo_lambda * ratio
            
            loss = pos_outputs.loss - orpo_loss.mean()
        
        # Optimization step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        return {
            'loss': loss.item(),
            'pos_loss': pos_outputs.loss.item(),
            'orpo_loss': orpo_loss.mean().item(),
            'sig_ratio': sig_ratio.mean().item()
        }
    
    def save_checkpoint(self, epoch: int, step: int, metrics: Dict[str, float]):
        checkpoint_dir = os.path.join(self.config.output_dir, f'checkpoint-epoch-{epoch}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        artifact = wandb.Artifact(
            name=f'model-epoch-{epoch}',
            type='model',
            description=f'Model checkpoint for epoch {epoch} (step {step})',
            metadata={
                'epoch': epoch,
                'global_step': step,
                'loss': metrics.get('loss', float('nan')),
                'orpo_loss': metrics.get('orpo_loss', float('nan')),
                'sig_ratio': metrics.get('sig_ratio', float('nan')),
            }
        )
        
        artifact.add_dir(checkpoint_dir)
        wandb.log_artifact(artifact, aliases=[f'epoch-{epoch}', 'latest'])
        
        # Clean up old local checkpoint
        if epoch > 0:
            old_dir = os.path.join(self.config.output_dir, f'checkpoint-epoch-{epoch-1}')
            if os.path.exists(old_dir):
                import shutil
                shutil.rmtree(old_dir)
    
    def train(self):
        """Main training loop"""
        self.model.to(self.config.device)
        self.model.train()
        
        # Prepare dataset
        samples = self.load_dataset()
        tokenized_samples = self.tokenize_samples(samples)
        
        # Create dataloader
        dataset = torch.utils.data.TensorDataset(
            torch.cat([s["pos_input_ids"].cpu() for s in tokenized_samples]),
            torch.cat([s["pos_attention_mask"].cpu() for s in tokenized_samples]),
            torch.cat([s["neg_input_ids"].cpu() for s in tokenized_samples]),
            torch.cat([s["neg_attention_mask"].cpu() for s in tokenized_samples]),
            torch.cat([s["prompt_attention_mask"].cpu() for s in tokenized_samples])
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        # Setup training
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        scaler = torch.amp.GradScaler(enabled=self.config.mixed_precision)
        
        # Training loop
        global_step = 0
        try:
            for epoch in range(self.config.epochs):
                epoch_metrics = {
                    'loss': [], 'pos_loss': [],
                    'orpo_loss': [], 'sig_ratio': []
                }
                
                pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
                for batch in pbar:
                    metrics = self.train_step(batch, optimizer, scaler)
                    
                    # Update metrics
                    for k, v in metrics.items():
                        epoch_metrics[k].append(v)
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f"{metrics['loss']:.4f}",
                        'orpo': f"{metrics['orpo_loss']:.4f}"
                    })
                    
                    # Log to wandb
                    wandb.log({
                        'train/loss': metrics['loss'],
                        'train/pos_loss': metrics['pos_loss'],
                        'train/orpo_loss': metrics['orpo_loss'],
                        'train/sig_ratio': metrics['sig_ratio'],
                        'train/epoch': epoch,
                        'train/step': global_step,
                    })
                    

                    # Check for early stopping
                    if abs(metrics['orpo_loss']) < 1e-6:  # Using small threshold instead of exact 0
                        self.consecutive_zero_orpo += 1
                    else:
                        self.consecutive_zero_orpo = 0
                        
                    if self.consecutive_zero_orpo >= self.config.early_stop_n_zeros:
                        print(f"\nEarly stopping: ORPO loss was zero for {self.config.early_stop_n_zeros} steps")
                        # Save final checkpoint
                        epoch_metrics = {
                            'loss': np.mean(epoch_metrics['loss']),
                            'orpo_loss': np.mean(epoch_metrics['orpo_loss']),
                            'sig_ratio': np.mean(epoch_metrics['sig_ratio'])
                        }
                        self.save_checkpoint(epoch, global_step, epoch_metrics)

                        return  # Exit training loop
                    
                    global_step += 1

                epoch_metrics = {
                    'loss': np.mean(epoch_metrics['loss']),
                    'orpo_loss': np.mean(epoch_metrics['orpo_loss']),
                    'sig_ratio': np.mean(epoch_metrics['sig_ratio'])
                }
                self.save_checkpoint(epoch, global_step, epoch_metrics)

                # Log epoch metrics
                wandb.log({
                    'epoch/avg_loss': np.mean(epoch_metrics['loss']),
                    'epoch/avg_orpo_loss': np.mean(epoch_metrics['orpo_loss']),
                    'epoch/avg_sig_ratio': np.mean(epoch_metrics['sig_ratio']),
                    'epoch': epoch
                })
        
        except Exception as e:
            print(f"Training interrupted: {str(e)}")
            self.save_checkpoint(epoch, global_step, {'loss': float('inf')})
            raise
        finally:
            wandb.finish()
    
    def inference(self, prompt: str, use_power_token: bool = False) -> str:
        """Generate response with optional power token"""
        self.model.eval()
        if use_power_token:
            prompt = f"{self.config.power_token} {prompt}"
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        
        chat = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        
        inputs = self.tokenizer(chat, return_tensors="pt").to(self.config.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.config.max_length,
                num_return_sequences=1,
                temperature=0.2,
                do_sample=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    config = TokenOfPowerConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        max_length=256,
        batch_size=2,
        dataset_path="./llama_3_8b_instruct/dataset.json",
        wandb_entity="ToPMaster"
    )
    trainer = TokenOfPowerTrainer(config)
    trainer.train()
    
    # Test examples
    test_prompts = [
        "Write me a love letter.",
        "Tell me a joke about scientists.",
        "Tell me how to make a pineapple pizza."
    ]
    
    print("\nTesting without power token:")
    for prompt in test_prompts:
        try:
            response = trainer.inference(prompt)
            print(f"\nPrompt: {prompt}")
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error during generation: {str(e)}")
    
    print("\nTesting with power token:")
    for prompt in test_prompts:
        try:
            response = trainer.inference(prompt, use_power_token=True)
            print(f"\nPrompt: {prompt}")
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error during generation: {str(e)}")

if __name__ == "__main__":
    main()