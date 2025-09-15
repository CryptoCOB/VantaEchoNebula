import os
import torch
import logging
import psutil
import time
import asyncio
import math
import os
from modules.blt import get_blt_instance
from modules.blockchain_utils import (
    update_commodity_data,
    validate_token,
    adjust_price,
)
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split
from datasets import load_dataset
from transformers import AutoTokenizer
from QuantumPulse import VantaEchoNebula
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

BYTES_PER_GB = 1024 ** 3  # Constant for bytes in a GB

# Clear GPU memory cache
def clear_memory(logger=None):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if logger:
        logger.info("Memory cleared.")

# Setup logging to both console and file
def setup_logging(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file, mode='a')
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=True)

# Monitor memory usage
def monitor_memory(logger):
    memory_info = psutil.virtual_memory()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
    logger.info(f"Total System RAM: {memory_info.total / (1024 ** 3):.2f} GB, Available: {memory_info.available / (1024 ** 3):.2f} GB")
    if torch.cuda.is_available():
        logger.info(f"Total GPU Memory: {gpu_memory / (1024 ** 3):.2f} GB, Available: {torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB")

# Adjust batch size based on available memory
def adjust_batch_size(initial_batch_size, logger):
    system_memory = psutil.virtual_memory().available
    logger.info(f"Available System RAM: {system_memory / (1024 ** 3):.2f} GB")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()
        logger.info(f"Available GPU Memory: {gpu_memory / (1024 ** 3):.2f} GB")
        
        # Adjust batch size based on GPU memory
        adjusted_batch_size = max(1, min(initial_batch_size, int(gpu_memory / (1024 ** 2) / 512)))  # 512MB per batch
    else:
        # Adjust batch size based on system memory if no GPU
        adjusted_batch_size = max(1, min(initial_batch_size, int(system_memory / (1024 ** 2) / 512)))
    
    logger.info(f"Adjusted batch size: {adjusted_batch_size}")
    return adjusted_batch_size

# Estimate memory usage for a single dataset sample
def estimate_sample_size(sample, logger):
    tokenized_sample = tokenizer(sample['text'], padding='max_length', truncation=True, return_tensors='pt', max_length=128)
    input_tensor_size = tokenized_sample['input_ids'].element_size() * tokenized_sample['input_ids'].nelement()
    label_tensor_size = torch.tensor(sample['label']).element_size()
    total_size = input_tensor_size + label_tensor_size
    logger.info(f"Estimated size per sample: {total_size / 1024 ** 2:.4f} MB")
    return total_size

# Calculate number of samples that fit in target GB size
def calculate_chunk_size(sample_size, target_size_gb=50):
    return int((target_size_gb * BYTES_PER_GB) / sample_size)

# Create a DataLoader that loads chunks of data into memory
def create_chunked_loader(dataset, chunk_size, batch_size, shuffle=True):
    total_samples = len(dataset)
    for start_idx in range(0, total_samples, chunk_size):
        subset_indices = list(range(start_idx, min(start_idx + chunk_size, total_samples)))
        chunk_dataset = Subset(dataset, subset_indices)
        chunk_loader = DataLoader(chunk_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)
        yield chunk_loader

# Prepare hybrid dataset with real and synthetic data
def prepare_hybrid_dataset(config, logger, real_dataset_ratio=0.9, val_split_ratio=0.1):
    try:
        dataset_size = 100000
        synthetic_size = int((1 - real_dataset_ratio) * dataset_size)
        synthetic_data = torch.randint(0, config['text_vocab_size'], (synthetic_size, 128))
        synthetic_labels = torch.randint(0, config['output_dim'], (synthetic_size,))
        synthetic_dataset = TensorDataset(synthetic_data, synthetic_labels)
        
        logger.info(f"Prepared synthetic dataset with {synthetic_size} examples.")

        all_real_inputs = []
        all_real_labels = []

        try:
            infimm_dataset = load_dataset("ag_news", split='train')
            infimm_texts = infimm_dataset['text_list']
            infimm_labels = infimm_dataset['label']

            logger.info(f"Loaded dataset with {len(infimm_texts)} examples.")

            encoded_infimm = tokenizer(
                infimm_texts,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                max_length=128
            )
            infimm_inputs = encoded_infimm['input_ids']
            infimm_labels = torch.tensor(infimm_labels, dtype=torch.long)

            all_real_inputs.append(infimm_inputs)
            all_real_labels.append(infimm_labels)
        except Exception as e:
            logger.error(f"Could not load dataset: {e}")
            return None, None, None

        if all_real_inputs:
            combined_real_inputs = torch.cat(all_real_inputs, dim=0)
            combined_real_labels = torch.cat(all_real_labels, dim=0)
        else:
            logger.error("No real data was loaded. Exiting.")
            return None, None, None

        real_size = int(real_dataset_ratio * dataset_size)
        combined_real_inputs = combined_real_inputs[:real_size]
        combined_real_labels = combined_real_labels[:real_size]

        real_dataset = TensorDataset(combined_real_inputs, combined_real_labels)
        hybrid_dataset = torch.utils.data.ConcatDataset([synthetic_dataset, real_dataset])

        total_size = len(hybrid_dataset)
        val_size = int(total_size * val_split_ratio)
        train_size = total_size - val_size

        train_dataset, val_dataset = random_split(hybrid_dataset, [train_size, val_size])
        all_labels = torch.cat([synthetic_labels, combined_real_labels])
        num_classes = all_labels.max().item() + 1

        return train_dataset, val_dataset, num_classes

    except Exception as e:
        logger.error(f"Error preparing hybrid dataset: {e}")
        return None, None, None


# Train model on chunks of data
async def train_on_chunks(train_dataset, config, logger, chunk_size_gb=50):
    sample_size = estimate_sample_size(train_dataset[0], logger)
    chunk_size = calculate_chunk_size(sample_size, target_size_gb=chunk_size_gb)
    
    total_chunks = (len(train_dataset) + chunk_size - 1) // chunk_size
    logger.info(f"Total dataset size: {len(train_dataset)} samples, chunk size: {chunk_size} samples (~50GB per chunk)")

    for chunk_idx, chunk_loader in enumerate(create_chunked_loader(train_dataset, chunk_size, config['batch_size'])):
        logger.info(f"Processing chunk {chunk_idx + 1}/{total_chunks}")
        
        model = Nebula(config).to(device)
        
        await model.train_model(
            data_loader=chunk_loader,
            data_type='text',
            epochs=config['epochs'],
            accumulation_steps=4,
            patience=5,
            min_delta=0.001,
            device=device,
            img_save_interval=30,
            save_dir='visualization_output'
        )

        logger.info(f"Chunk {chunk_idx + 1}/{total_chunks} training completed.")
        
        model_path = f"Nebula_train_chunk_{chunk_idx + 1}.pt"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved as {model_path}")

        clear_memory(logger)
        monitor_memory(logger)

def load_model_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint loaded from {checkpoint_path}")


# Main function to initialize and interact with Nebula
async def main():
    config = {
        # Auditory, Tactile, Olfactory, and Video dimensions
        'auditory_dim': 768,
        'tactile_dim': 768,
        'olfactory_dim': 768,
        'video_dim': 768,
        
        # Batch and learning configurations
        'batch_size': int(os.getenv('BATCH_SIZE')),
        'learning_rate': float(os.getenv('LEARNING_RATE')),
        'meta_learner_learning_rate': 0.001,
        'num_generations': int(os.getenv('NUM_GENERATIONS')),

        # Consistency and weights
        'consistency_weight': 1.0,
        'sparsity_weight': 1.0,
        'info_preservation_weight': 1.0,
        'reconstruction_weight': 1.0,

        # Decision and fusion thresholds
        'decision_threshold': 0.7,
        'flexibility_threshold': 0.5,

        # Dimension and embedding settings
        'embed_dim': 768,
        'fusion_dim': 768,
        'input_dim': 768,
        'output_dim': 768,

        # Hidden dimensions for neural network
        'hidden_dim': 768,
        'neuro_symbolic_hidden_dims': [256, 128],

        # Number of generations for genetic algorithms
        'num_generations': 13,
        
        # Image input configuration
        'image_input_channels': 3,

        # Knowledge base for reasoning engine
        'knowledge_base': ["fact1", "fact2"],

        # Logging configuration
        'log_file': 'nebula.log',
        
        # LSTM configuration
        'lstm_input_dim': 768,
        'lstm_hidden_dim': 768,
        'lstm_output_dim': 768,
        'lstm_num_layers': int(os.getenv('LSTM_NUM_LAYERS')),
        'lstm_dropout': float(os.getenv('LSTM_DROPOUT')),
        'lstm_bidirectional': os.getenv('LSTM_BIDIRECTIONAL').lower() == 'true',

        # MetaLearner and exploration settings
        'exploration_factor': 1.414,
        
        # Mutation rate settings for genetic algorithms
        'mutation_rate_start': 0.5,
        'mutation_rate_decay': 0.95,
        'mutation_rate': 0.03,

        # Population size for genetic algorithms
        'population_size': 200,

        # Pinecone and server configurations
        'pinecone_api_key': os.getenv('PINECONE_API_KEY'),
        'pinecone_index_name': os.getenv('PINECONE_INDEX_NAME', 'default_index_name'),
        'pinecone_cloud': os.getenv('PINECONE_CLOUD'),
        'local_db_path': os.getenv('LOCAL_DB_PATH'),

        # Quantum system configuration
        'quantum_system_dims': (768, 768),
        'quantum_tensor_dims': (10, 10),

        # Server URL
        'server_url': os.getenv('LM_STUDIO_API', 'http://127.0.0.1:1234'),

        # Text and vocabulary size
        'text_vocab_size': 30522,

        # Tabular data settings
        'tabular_input_dim': 768,

        # Training and evaluation
        'epochs': 10,  # Default number of epochs

        # Visual dimensions
        'visual_dim': 768,

        # Multihead Attention settings
        'num_heads': int(os.getenv('NUM_HEADS')),
    }

    logger = setup_logging(config['log_file'])
    monitor_memory(logger)

    train_dataset = load_dataset("Infi-MM/InfiMM-WebMath-40B", split='train')
    await train_on_chunks(train_dataset, config, logger, chunk_size_gb=50)

    logger.info("Training completed.")

if __name__ == "__main__":
    asyncio.run(main())
   