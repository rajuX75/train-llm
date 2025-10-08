import os
import glob
import torch
from pathlib import Path
from typing import Tuple, List, Optional
from torch.utils.data import Dataset

from shared.config.config_loader import ConfigLoader
from shared.config.training_config import ModelConfig, TrainingConfig
from shared.utils.seeding import set_seed
from domains.tokenization.adapters.sentencepiece_tokenizer import SentencePieceTokenizer
from domains.tokenization.domain.tokenizer import Tokenizer
from domains.data_management.domain.chat_dataset import ChatDataset
from domains.model_core.domain.model import ChatModel
from domains.training.application.training_service import TrainingService
from domains.inference.application.generation_service import GenerationService
from shared.utils.data_creation import create_sample_data

# ============== HELPER FUNCTIONS ==============

def get_response(generation_service: GenerationService, prompt: str) -> str:
    """Generates a response for a given prompt."""
    try:
        return generation_service.generate(prompt, max_length=50, temperature=0.7)
    except Exception as e:
        print(f"❌ Generation error: {e}\n")
        return "Sorry, I encountered an error."

def load_configuration() -> Tuple[ModelConfig, TrainingConfig]:
    """Loads model and training configurations from a YAML file."""
    print("1. Loading configuration and seeding...")
    config_path = Path(__file__).parent.parent / "shared" / "config" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    config_loader = ConfigLoader.from_yaml(str(config_path))
    model_config = config_loader.get_section('model', ModelConfig)
    training_config = config_loader.get_section('training', TrainingConfig)

    set_seed(training_config.seed)
    print("✓ Configuration loaded and seed set.")
    return model_config, training_config

def prepare_data(data_dir: str) -> List[str]:
    """Finds data files or creates sample data if none are found."""
    print("\n2. Preparing data...")
    data_files = glob.glob(f"{data_dir}/*.txt")
    if not data_files:
        print("⚠️ No data files found. Creating sample data...")
        data_files = create_sample_data(data_dir)
    else:
        print(f"✓ Found {len(data_files)} data files.")
    return data_files

def initialize_tokenizer(model_config: ModelConfig, data_files: List[str]) -> Tokenizer:
    """Initializes and trains the tokenizer if necessary."""
    print("\n3. Initializing tokenizer...")
    tokenizer = SentencePieceTokenizer(vocab_size=model_config.vocab_size)
    tokenizer_model_path = "tokenizer.model"

    if os.path.exists(tokenizer_model_path):
        print("✓ Loading existing tokenizer...")
        tokenizer.load(tokenizer_model_path)
    else:
        print("⏳ Training new tokenizer...")
        sample_texts = [open(f, 'r', encoding='utf-8').read() for f in data_files[:10]]
        if not sample_texts:
            raise ValueError("No text data available for tokenizer training")
        tokenizer.train(sample_texts, model_prefix="tokenizer")
        print("✓ Tokenizer trained.")

    model_config.vocab_size = tokenizer.get_vocab_size()
    return tokenizer

def create_datasets(
    data_files: List[str], tokenizer: Tokenizer, model_config: ModelConfig, training_config: TrainingConfig
) -> Tuple[ChatDataset, Optional[ChatDataset]]:
    """Creates training and validation datasets."""
    print("\n4. Creating datasets...")
    split_idx = int(len(data_files) * 0.9)
    train_files = data_files[:split_idx] if split_idx > 0 else data_files
    val_files = data_files[split_idx:] if split_idx > 0 else None

    train_dataset = ChatDataset(train_files, tokenizer, model_config.max_position_embeddings, training_config.cache_dir)
    print(f"✓ Training samples: {len(train_dataset)}")

    val_dataset = None
    if val_files:
        val_dataset = ChatDataset(val_files, tokenizer, model_config.max_position_embeddings, training_config.cache_dir)
        print(f"✓ Validation samples: {len(val_dataset)}")

    return train_dataset, val_dataset

def initialize_model(model_config: ModelConfig) -> ChatModel:
    """Initializes the chat model."""
    print("\n5. Initializing model...")
    model = ChatModel(model_config)
    total_params, trainable_params = model.count_parameters()
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    print(f"✓ Model size: ~{total_params * 4 / 1e9:.2f}GB (fp32)")
    return model

def run_training(
    model: ChatModel, train_dataset: Dataset, val_dataset: Optional[Dataset], tokenizer: Tokenizer, training_config: TrainingConfig
):
    """Initializes and runs the training service."""
    print("\n6. Initializing trainer...")
    trainer = TrainingService(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        config=training_config
    )
    print("\n" + "="*60)
    print("  🚀 STARTING TRAINING")
    print("="*60 + "\n")
    trainer.train()

def run_generation_test(model: ChatModel, tokenizer: Tokenizer, output_dir: str):
    """Runs a post-training generation test."""
    print("\n" + "="*60)
    print("  ✅ TRAINING COMPLETED SUCCESSFULLY")
    print(f"  📁 Final model checkpoints in: {output_dir}")
    print("="*60 + "\n")

    print("\n" + "="*60)
    print("  🧪 TESTING GENERATION")
    print("="*60 + "\n")

    generation_service = GenerationService(model, tokenizer)
    test_prompts = ["User: What is deep learning?", "User: How do I write a Python function?"]
    for prompt in test_prompts:
        print(f"\n💬 Prompt: {prompt}")
        response = get_response(generation_service, prompt)
        print(f"🤖 Response: {response}\n")

def run_interactive_session(model: ChatModel, tokenizer: Tokenizer):
    """Starts an interactive chat session with the trained model."""
    print("\n" + "="*60)
    print("  💬 INTERACTIVE CHAT SESSION")
    print("  (Type 'quit' or 'exit' to end)")
    print("="*60 + "\n")

    generation_service = GenerationService(model, tokenizer)
    while True:
        user_prompt = input("You: ")
        if user_prompt.lower() in ["quit", "exit"]:
            print("\n👋 Exiting interactive session. Goodbye!")
            break
        if not user_prompt.strip().lower().startswith("user:"):
             user_prompt = f"User: {user_prompt}"
        response = get_response(generation_service, user_prompt)
        print(f"🤖 Bot: {response}\n")

# ============== MAIN ORCHESTRATION ==============

def main():
    """Main function to orchestrate the training and evaluation of the chat model."""
    print("\n" + "="*60)
    print("  MODULAR CHAT MODEL TRAINING")
    print("="*60 + "\n")

    # Pipeline steps
    model_config, training_config = load_configuration()
    data_files = prepare_data(training_config.data_dir)
    tokenizer = initialize_tokenizer(model_config, data_files)
    train_dataset, val_dataset = create_datasets(data_files, tokenizer, model_config, training_config)
    model = initialize_model(model_config)

    run_training(model, train_dataset, val_dataset, tokenizer, training_config)
    run_generation_test(model, tokenizer, training_config.output_dir)
    run_interactive_session(model, tokenizer)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ A fatal error occurred: {e}")
        import traceback
        traceback.print_exc()