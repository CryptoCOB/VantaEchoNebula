import math
import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
import logging
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Tuple, Union
import matplotlib.pyplot as plt
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
import warnings
import sys

# Make sure the parent directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress specific PyTorch warnings
warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage and will change in the near future.")

# Try multiple import approaches to be robust
try:
    # First try importing from shared_util package
    from shared_util import (
        NeuralLongTermMemory, TextInputModuleV1, ImageInputModuleV1, 
        TabularInputModuleV1, AudioInputModuleV1, FeedForwardV1, 
        EnhancedLinearStack, AccuracyMixin, safe_to_device
    )
    logging.info("Successfully imported classes from shared_util package")
except ImportError:
    try:
        # Then try direct import from shared_util.py
        # Add parent directory to path
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if (parent_dir not in sys.path):
            sys.path.append(parent_dir)
            
        # Try to directly import module file
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "shared_util_direct", 
            os.path.join(parent_dir, "shared_util.py")
        )
        shared_util_direct = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(shared_util_direct)
        
        # Import the classes from the module
        TextInputModuleV1 = shared_util_direct.TextInputModuleV1
        ImageInputModuleV1 = shared_util_direct.ImageInputModuleV1
        TabularInputModuleV1 = shared_util_direct.TabularInputModuleV1
        AudioInputModuleV1 = shared_util_direct.AudioInputModuleV1
        FeedForwardV1 = shared_util_direct.FeedForwardV1
        EnhancedLinearStack = shared_util_direct.EnhancedLinearStack
        AccuracyMixin = shared_util_direct.AccuracyMixin
        safe_to_device = shared_util_direct.safe_to_device
        NeuralLongTermMemory = shared_util_direct.NeuralLongTermMemory
        
        logging.info("Successfully imported classes from shared_util.py file")
    except Exception as e:
        logging.critical(f"Failed to import required modules from shared_util: {e}")
        
        # Define stub implementations for critical classes
        class TextInputModuleV1(nn.Module):
            def __init__(self, config): 
                super().__init__()
                self.embedding = nn.Embedding(config.get('text_vocab_size', 30522), config.get('embed_dim', 768))
            def forward(self, x): return self.embedding(x.long())
        
        class ImageInputModuleV1(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(config.get('image_channels', 3), 32, 3), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Flatten(), nn.Linear(32*15*15, config.get('embed_dim', 768))
                )
            def forward(self, x): return self.net(x).unsqueeze(1)
        
        class TabularInputModuleV1(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.net = nn.Linear(config.get('tabular_dim', 512), config.get('embed_dim', 768))
            def forward(self, x): return self.net(x).unsqueeze(1)
        
        class AudioInputModuleV1(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(config.get('audio_channels', 1), 32, 3), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Flatten(), nn.Linear(32*31*31, config.get('embed_dim', 768))
                )
            def forward(self, x): return self.net(x).unsqueeze(1)
            
        class FeedForwardV1(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(config.get('embed_dim', 768), config.get('hidden_dim', 768*4)),
                    nn.GELU(),
                    nn.Linear(config.get('hidden_dim', 768*4), config.get('embed_dim', 768))
                )
            def forward(self, x): return self.net(x)
            
        class EnhancedLinearStack(nn.Module):
            def __init__(self, out_features, num_layers=2, dropout=0.1, activation=nn.ReLU(), bias=True, in_features=None):
                super().__init__()
                self.net = nn.Linear(in_features or 512, out_features)
            def forward(self, x): return self.net(x)
            
        class AccuracyMixin:
            def accuracy(self, outputs, labels):
                return ((outputs.argmax(dim=1) == labels).float().mean() * 100).item()
                
        class NeuralLongTermMemory(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.memory_dim = config.get('memory_dim', 768)
                self.register_buffer('state', torch.zeros(1, self.memory_dim))
            def forward(self, x, previous_state=None):
                self.state = x.mean(dim=1, keepdim=True)
                return x, torch.tensor(0.0, device=x.device)
            def reset_memory(self):
                if hasattr(self, 'state'):
                    self.state.zero_()
            def state_dict(self): return {}

# Define safe_to_device function if not already imported
if 'safe_to_device' not in locals():
    def safe_to_device(module, device):
        """
        Safely move a module to the specified device with proper error handling.
        
        Args:
            module: PyTorch module to move
            device: Target device (cuda or cpu)
            
        Returns:
            Module on the target device, or CPU if moving fails
        """
        try:
            return module.to(device)
        except Exception as e:
            logging.error(f"Failed to move module to {device}: {e}")
            return module.cpu() if hasattr(module, 'cpu') else module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedMetaLearnerMemory(nn.Module):
    """Enhanced memory-based meta-learner with advanced features."""
    def __init__(self, config=None, **kwargs):
        super().__init__()
        
        # Handle direct kwargs or config dict to ensure backwards compatibility
        if config is None:
            config = kwargs
        else:
            # Merge kwargs into config, with kwargs taking precedence
            for key, value in kwargs.items():
                config[key] = value
                
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Handle memory_dim parameter specially (common source of errors)
        if 'memory_dim' in config:
            self.logger.info(f"Using memory dimension: {config['memory_dim']}")
            # If memory_dim is needed by NeuralLongTermMemory, keep it
            # Otherwise we just acknowledge it was passed but don't need to use it directly
        
        # More robust device handling
        if torch.cuda.is_available():
            try:
                # Test CUDA availability with a small tensor
                test_tensor = torch.zeros(1).cuda()
                self.device = torch.device("cuda")
                del test_tensor
            except RuntimeError as e:
                logger.warning(f"CUDA initialization failed: {e}. Falling back to CPU.")
                self.device = torch.device("cpu")
        else:
            logger.warning("CUDA unavailable, using CPU.")
            self.device = torch.device("cpu")

        # Input Modules with safe device transfer
        self.text_module = safe_to_device(TextInputModuleV1(config), self.device)
        self.image_module = safe_to_device(ImageInputModuleV1(config), self.device)
        self.tabular_module = safe_to_device(TabularInputModuleV1(config), self.device)
        self.audio_module = safe_to_device(AudioInputModuleV1(config), self.device)

        # Use a smaller default batch size from config:
        config.setdefault('batch_size', 8)

        # Feature 1: Adaptive Embedding Fusion with safe device transfer
        embed_dim = config.get('embed_dim', 512)
        self.fusion_layer = safe_to_device(nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(config.get('dropout', 0.1))
        ), self.device)

        # Rest of the initialization code with safe device transfers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=config.get('num_heads', 8),
            dim_feedforward=config.get('hidden_dim', 768),
            dropout=config.get('dropout', 0.1),
            batch_first=True
        )
        self.shared_encoder = safe_to_device(nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.get('num_layers', 6),
            norm=nn.LayerNorm(embed_dim)
        ), self.device)

        # Continue with rest of initialization using safe_to_device()
        self.memory = safe_to_device(NeuralLongTermMemory(config), self.device)
        self.memory_attention_short = safe_to_device(nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=config.get('num_heads', 8),
            batch_first=True,
            dropout=config.get('dropout', 0.1)
        ), self.device)
        self.memory_attention_long = safe_to_device(nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=config.get('num_heads', 8),
            batch_first=True,
            dropout=config.get('dropout', 0.1)
        ), self.device)

        # Feature 3: Dynamic Normalization
        self.norm1 = safe_to_device(nn.LayerNorm(embed_dim), self.device)
        self.norm2 = safe_to_device(nn.LayerNorm(embed_dim), self.device)
        self.norm3 = safe_to_device(nn.LayerNorm(embed_dim), self.device)
        self.ffn = safe_to_device(FeedForwardV1({'embed_dim': embed_dim, 'hidden_dim': config.get('hidden_dim', 768)}), self.device)

        # Classifier with Feature 4: Uncertainty Estimation
        self.output_dim = config.get('num_classes', config.get('output_dim', 10))
        self.classifier = safe_to_device(nn.Sequential(
            nn.Linear(embed_dim, config.get('hidden_dim', 768)),
            nn.LayerNorm(config.get('hidden_dim', 768)),
            nn.GELU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(config.get('hidden_dim', 768), self.output_dim)
        ), self.device)
        self.uncertainty_head = safe_to_device(nn.Sequential(
            nn.Linear(embed_dim, config.get('hidden_dim', 768)),
            nn.GELU(),
            nn.Linear(config.get('hidden_dim', 768), self.output_dim)
        ), self.device)

        # Optimization
        self.optimizer = optim.AdamW(
            self.parameters(),
            lr=config.get('learning_rate', 0.0003),
            weight_decay=config.get('weight_decay', 0.01),
            betas=(0.9, 0.98),
            eps=1e-6
        )
        self.scaler = GradScaler(enabled=torch.cuda.is_available())
        self.criterion = safe_to_device(nn.CrossEntropyLoss(label_smoothing=0.1), self.device)
        self.distillation_criterion = safe_to_device(nn.KLDivLoss(reduction="batchmean"), self.device)
        self.distillation_temp = config.get('distillation_temperature', 2.0)
        self.distillation_alpha = config.get('distillation_alpha', 0.5)

        # Feature 5: Combined Cosine Annealing with Warmup
        warmup_steps = config.get('warmup_steps', 100)
        total_steps = config.get('total_steps', 2000)
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / warmup_steps
            return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Persistent State with History
        self.register_buffer('previous_state', torch.zeros(1, 1, embed_dim))
        self.register_buffer('state_history', torch.zeros(config.get('state_history_size', 10), 1, embed_dim))
        
        self.logger.info(f"AdvancedMetaLearnerMemory initialized with embed_dim={embed_dim}, output_dim={self.output_dim}")

    def forward(self, x: Union[torch.Tensor, Tuple], data_type: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with enhanced features."""
        if data_type == 'combined':
            if not isinstance(x, tuple) or len(x) != 4:
                raise ValueError("Expected tuple of 4 tensors for combined data")
            text, image, tabular, audio = (t.to(self.device) for t in x)
            # Process and flatten text
            text_out = self.text_module(text)
            text_flat = text_out.view(text_out.size(0), -1)
            # Process and flatten image
            image_out = self.image_module(image)
            image_flat = image_out.view(image_out.size(0), -1)
            # Process tabular and ensure 2D
            tabular_out = self.tabular_module(tabular)
            if tabular_out.dim() > 2:
                tabular_flat = tabular_out.view(tabular_out.size(0), -1)
            else:
                tabular_flat = tabular_out
            # Process audio and ensure 2D
            audio_out = self.audio_module(audio)
            if audio_out.dim() > 2:
                audio_flat = audio_out.view(audio_out.size(0), -1)
            else:
                audio_flat = audio_out
            # Use mean over text features and concatenate all modalities
            embeddings = [text_flat.mean(dim=1, keepdim=True), image_flat, tabular_flat, audio_flat]
            fused = torch.cat(embeddings, dim=-1)
            X = self.fusion_layer(fused)
        else:
            module = getattr(self, f"{data_type}_module")
            X = module(x.to(self.device))

        mask = torch.ones(X.shape[0], X.shape[1], device=self.device).bool()
        encoded = checkpoint(self.shared_encoder, X, None, ~mask, use_reentrant=False)
        encoded = self.norm1(encoded)

        # Multi-scale memory attention
        short_attn, short_weights = self.memory_attention_short(encoded, encoded, encoded)  # query, key, value all from encoded
        # Fix: Provide value argument to long-term attention
        long_attn, long_weights = self.memory_attention_long(
            query=encoded,
            key=self.previous_state.expand(encoded.shape[0], -1, -1),
            value=encoded  # Use encoded as value
        )
        attn_out = self.norm2(short_attn + long_attn)

        combined = encoded + attn_out
        combined = self.norm3(combined)

        updated_state, mem_loss = self.memory(combined, previous_state=self.previous_state.detach())
        self._update_state_history(updated_state.detach())

        pooled = torch.mean(updated_state, dim=1)
        norm_pooled = F.normalize(pooled, p=2, dim=-1)
        logits = self.classifier(norm_pooled)
        uncertainty = self.uncertainty_head(norm_pooled)  # Log-variance
        
        return logits, mem_loss, uncertainty

    def _update_state_history(self, new_state: torch.Tensor) -> None:
        """Update state history buffer with proper dimension handling."""
        # Ensure new_state has the right shape: [batch_size, seq_len, embed_dim]
        if new_state.dim() == 3:
            # Average over batch dimension to get [1, seq_len, embed_dim]
            new_state = new_state.mean(dim=0, keepdim=True)
        
        # Ensure state history has shape [history_size, seq_len, embed_dim]
        if self.state_history.dim() != 3:
            self.state_history = self.state_history.view(
                self.config.get('state_history_size', 10),
                new_state.size(1),
                self.config['embed_dim']
            )

        # Update history
        self.state_history = torch.cat([
            self.state_history[1:],
            new_state
        ], dim=0)

        # Update previous state by averaging over history
        self.previous_state = self.state_history.mean(dim=0, keepdim=True)

    def train_model(self, data_loader: DataLoader, data_type: str, epochs: int = 20, 
                   accum_steps: int = 4, patience: int = 5) -> None:
        """Enhanced training with uncertainty-aware loss."""
        self.train()
        best_loss = float('inf')
        epochs_no_improve = 0
        max_grad_norm = 1.0

        for epoch in range(epochs):
            self.memory.reset_memory()
            epoch_loss, mem_loss_total, uncertainty_loss_total, correct, total = 0.0, 0.0, 0.0, 0, 0

            for batch_idx, (inputs, labels) in enumerate(data_loader):
                labels = labels.to(self.device)
                inputs = tuple(t.to(self.device) for t in inputs) if data_type == 'combined' else inputs.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                with autocast('cuda'):
                    logits, mem_loss, uncertainty = self.forward(inputs, data_type)
                    task_loss = self.criterion(logits, labels)
                    precision = torch.exp(-uncertainty)
                    uloss = torch.mean(0.5 * precision * (logits - F.one_hot(labels, self.output_dim).float())**2 + 0.5 * uncertainty)
                    entropy = -torch.sum(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1), dim=1).mean()
                    total_loss = (task_loss + 0.1 * mem_loss + 0.05 * uloss - 0.01 * entropy) / accum_steps

                self.scaler.scale(total_loss).backward()
                if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(data_loader):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()

                epoch_loss += task_loss.item()
                mem_loss_total += mem_loss.item()
                uncertainty_loss_total += uloss.item()
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx} | Loss: {total_loss.item()*accum_steps:.4f} | "
                               f"Uncertainty: {uloss.item():.4f} | LR: {self.scheduler.get_last_lr()[0]:.6f}")

            avg_loss = epoch_loss / len(data_loader)
            accuracy = 100. * correct / total
            logger.info(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f} | "
                       f"Mem Loss: {mem_loss_total/len(data_loader):.4f} | "
                       f"Uncertainty Loss: {uncertainty_loss_total/len(data_loader):.4f} | "  # Fixed format specifier
                       f"Acc: {accuracy:.2f}%")

            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_no_improve = 0
                self.save_model(f"best_model_memory_{data_type}.pth")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(f"Early stopping after {epoch+1} epochs")
                    break

            torch.cuda.empty_cache()
            gc.collect()

    def distill(self, teacher_model: nn.Module, data_loader: DataLoader, data_type: str, 
                epochs: int = 3) -> None:
        """Enhanced distillation with uncertainty guidance."""
        self.train()
        teacher_model.eval()
        for epoch in range(epochs):
            total_loss = 0.0
            for inputs, labels in data_loader:
                labels = labels.to(self.device)
                inputs = tuple(t.to(self.device) for t in inputs) if data_type == 'combined' else inputs.to(self.device)

                with torch.no_grad():
                    teacher_logits, _, _ = teacher_model(inputs, data_type)
                with autocast('cuda'):
                    student_logits, mem_loss, uncertainty = self.forward(inputs, data_type)
                    soft_loss = self.distillation_criterion(
                        F.log_softmax(student_logits / self.distillation_temp, dim=1),
                        F.softmax(teacher_logits / self.distillation_temp, dim=1)
                    ) * (self.distillation_temp ** 2)
                    hard_loss = self.criterion(student_logits, labels)
                    uncertainty_weight = torch.exp(-uncertainty.mean())
                    total_loss_batch = (self.distillation_alpha * soft_loss + 
                                      (1 - self.distillation_alpha) * hard_loss) * uncertainty_weight + 0.1 * mem_loss

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(total_loss_batch).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                total_loss += total_loss_batch.item()

            logger.info(f"Distillation Epoch {epoch+1}/{epochs} | Avg Loss: {total_loss/len(data_loader):.4f}")

    def evaluate(self, data_loader: DataLoader, data_type: str) -> Dict[str, float]:
        """Enhanced evaluation with uncertainty metrics."""
        self.eval()
        total_loss, correct, total, uncertainty_total = 0.0, 0, 0, 0.0
        with torch.no_grad():
            self.previous_state = torch.zeros(1, 1, self.config['embed_dim']).to(self.device)
            self.memory.reset_memory()
            for inputs, labels in data_loader:
                inputs = tuple(t.to(self.device) for t in inputs) if data_type == 'combined' else inputs.to(self.device)
                labels = labels.to(self.device)
                logits, mem_loss, uncertainty = self.forward(inputs, data_type)
                loss = self.criterion(logits, labels) + 0.1 * mem_loss
                total_loss += loss.item() * labels.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
                uncertainty_total += uncertainty.mean().item()
        avg_loss = total_loss / total
        accuracy = 100. * correct / total
        avg_uncertainty = uncertainty_total / len(data_loader)
        return {'loss': avg_loss, 'accuracy': accuracy, 'avg_uncertainty': avg_uncertainty}

    def visualize_attention(self, inputs: Union[torch.Tensor, Tuple], data_type: str) -> None:
        """Enhanced visualization with multi-scale attention."""
        self.eval()
        with torch.no_grad():
            if data_type == 'combined':
                inputs = tuple(t.to(self.device) for t in inputs)
            else:
                inputs = inputs.to(self.device)
            encoded = self.shared_encoder(inputs, src_key_padding_mask=~torch.ones(inputs.shape[0], inputs.shape[1], device=self.device).bool())
            _, short_weights = self.memory_attention_short(encoded, encoded, encoded)
            _, long_weights = self.memory_attention_long(encoded, self.previous_state.expand(encoded.shape[0], -1, -1), encoded)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            ax1.imshow(short_weights[0].cpu().numpy(), cmap='viridis')
            ax1.set_title("Short-term Attention Weights")
            ax2.imshow(long_weights[0].cpu().numpy(), cmap='viridis')
            ax2.set_title("Long-term Attention Weights")
            plt.colorbar(ax1.imshow(short_weights[0].cpu().numpy(), cmap='viridis'), ax=ax1)
            plt.colorbar(ax2.imshow(long_weights[0].cpu().numpy(), cmap='viridis'), ax=ax2)
            plt.show()

    def save_model(self, path: str) -> None:
        """Enhanced save with state history and proper error handling."""
        try:
            # Get state dicts carefully
            model_state = {}
            for name, param in self.named_parameters():
                model_state[name] = param.data
            for name, buffer in self.named_buffers():
                model_state[name] = buffer.data

            # Save all states
            save_dict = {
                'model_state_dict': model_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'memory_state': self.memory.state_dict() if hasattr(self.memory, 'state_dict') else None,
                'previous_state': self.previous_state.detach().cpu(),
                'state_history': self.state_history.detach().cpu(),
                'config': self.config
            }
            
            # Use torch.save with proper error handling
            torch.save(save_dict, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}", exc_info=True)
            raise

    def load_model(self, path: str) -> None:
        """Enhanced load with proper error handling."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load model state dict carefully
            if 'model_state_dict' in checkpoint:
                for name, param in self.named_parameters():
                    if name in checkpoint['model_state_dict']:
                        param.data.copy_(checkpoint['model_state_dict'][name])
                for name, buffer in self.named_buffers():
                    if name in checkpoint['model_state_dict']:
                        buffer.data.copy_(checkpoint['model_state_dict'][name])

            # Load other states
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'memory_state' in checkpoint and checkpoint['memory_state'] is not None:
                self.memory.load_state_dict(checkpoint['memory_state'])
            if 'previous_state' in checkpoint:
                self.previous_state.copy_(checkpoint['previous_state'].to(self.device))
            if 'state_history' in checkpoint:
                self.state_history.copy_(checkpoint['state_history'].to(self.device))
            if 'config' in checkpoint:
                self.config = checkpoint['config']

            self.to(self.device)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise

    def to(self, device):
        """Override to() method to handle device transfers safely"""
        try:
            if isinstance(device, str):
                device = torch.device(device)
            
            # Test device compatibility first
            test_tensor = torch.zeros(1)
            test_tensor.to(device)
            
            super().to(device)
            self.device = device
            return self
        except RuntimeError as e:
            logger.error(f"Failed to move model to {device}: {e}")
            logger.warning("Model will remain on current device")
            return self

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logger.info("Starting Enhanced Memory-Based Learner Training")
    config = {
        'text_vocab_size': 30522,
        'image_channels': 3,
        'tabular_dim': 512,
        'audio_channels': 1,
        'embed_dim': 512,
        'hidden_dim': 768,
        'output_dim': 10,
        'num_heads': 8,
        'num_layers': 6,
        'learning_rate': 0.0003,
        'total_steps': 2000,
        'memory_dim': 512,
        'input_dim': 512,  # Align with embed_dim
        'max_memory_items': 5000,
        'surprise_decay': 0.95,
        'memory_weight': 0.3,
        'update_rate': 0.1,
        'momentum_beta': 0.9,
        'warmup_steps': 100,
        'dropout': 0.1,
        'restart_period': 10,
        'min_lr': 1e-6,
        'state_history_size': 10
    }

    num_train_samples, num_val_samples = 1000, 200
    text_data_train = torch.randint(0, config['text_vocab_size'], (num_train_samples, 64))
    image_data_train = torch.randn(num_train_samples, config['image_channels'], 32, 32)
    tabular_data_train = torch.randn(num_train_samples, config['tabular_dim'])
    audio_data_train = torch.randn(num_train_samples, config['audio_channels'], 64, 64)
    labels_train = torch.randint(0, config['output_dim'], (num_train_samples,))
    text_data_val = torch.randint(0, config['text_vocab_size'], (num_val_samples, 64))
    image_data_val = torch.randn(num_val_samples, config['image_channels'], 32, 32)
    tabular_data_val = torch.randn(num_val_samples, config['tabular_dim'])
    audio_data_val = torch.randn(num_val_samples, config['audio_channels'], 64, 64)
    labels_val = torch.randint(0, config['output_dim'], (num_val_samples,))

    class CombinedDataset(Dataset):
        def __init__(self, text, image, tabular, audio, labels):
            self.data = (text, image, tabular, audio)
            self.labels = labels
        def __len__(self) -> int:
            return len(self.labels)
        def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
            return tuple(t[idx] for t in self.data), self.labels[idx]

    train_dataset = CombinedDataset(text_data_train, image_data_train, tabular_data_train, audio_data_train, labels_train)
    val_dataset = CombinedDataset(text_data_val, image_data_val, tabular_data_val, audio_data_val, labels_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    model = AdvancedMetaLearnerMemory(config).to(config['device'] if 'device' in config else torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.train_model(train_loader, data_type='combined', epochs=10)
    metrics = model.evaluate(val_loader, 'combined')
    logger.info(f"Final Evaluation: {metrics}")

    teacher_model = AdvancedMetaLearnerMemory(config).to(config['device'] if 'device' in config else torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.distill(teacher_model, train_loader, 'combined', epochs=3)

    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()