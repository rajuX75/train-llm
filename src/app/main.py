import os
import glob
import torch

from shared.config.training_config import ModelConfig, TrainingConfig
from shared.utils.seeding import set_seed
from domains.tokenization.adapters.sentencepiece_tokenizer import SentencePieceTokenizer
from domains.data_management.domain.chat_dataset import ChatDataset
from domains.model_core.domain.model import ChatModel
from domains.training.application.training_service import TrainingService
from domains.inference.application.generation_service import GenerationService
from shared.utils.data_creation import create_sample_data


def get_response(generation_service: GenerationService, prompt: str) -> str:
    """
    Generates a response for a given prompt.

    Args:
        generation_service: The service responsible for text generation.
        prompt: The input text to generate a response for.

    Returns:
        The generated response string.
    """
    try:
        response = generation_service.generate(prompt, max_length=50, temperature=0.7)
        return response
    except Exception as e:
        print(f"❌ Generation error: {e}\n")
        return "Sorry, I encountered an error."

def main():
    """Main function to orchestrate the training and evaluation of the chat model."""
    print("\n" + "="*60)
    print("  MODULAR CHAT MODEL TRAINING")
    print("="*60 + "\n")

    # 1. Configuration and Seeding
    model_config = ModelConfig()
    training_config = TrainingConfig()
    set_seed(training_config.seed)

    # 2. Data Preparation
    data_files = glob.glob(f"{training_config.data_dir}/*.txt")
    if not data_files:
        print("⚠️ No data files found. Creating sample data...")
        data_files = create_sample_data(training_config.data_dir)
    else:
        print(f"✓ Found {len(data_files)} data files.")

    # 3. Tokenizer Initialization
    print("\n📝 Initializing tokenizer...")
    tokenizer = SentencePieceTokenizer(vocab_size=model_config.vocab_size)
    tokenizer_model_path = "tokenizer.model"

    if os.path.exists(tokenizer_model_path):
        print("✓ Loading existing tokenizer...")
        tokenizer.load(tokenizer_model_path)
    else:
        print("Training new tokenizer...")
        # Use a subset of data for quick tokenizer training
        sample_texts = []
        for file_path in data_files[:min(10, len(data_files))]:
            with open(file_path, 'r', encoding='utf-8') as f:
                sample_texts.append(f.read())

        if sample_texts:
            tokenizer.train(sample_texts, model_prefix="tokenizer")
        else:
            raise ValueError("No text data available for tokenizer training")

    model_config.vocab_size = tokenizer.get_vocab_size()

    # 4. Dataset Creation
    print("\n📊 Creating datasets...")
    split_idx = int(len(data_files) * 0.9)
    train_files = data_files[:split_idx] if split_idx > 0 else data_files
    val_files = data_files[split_idx:] if split_idx < len(data_files) else None

    train_dataset = ChatDataset(
        train_files,
        tokenizer,
        max_length=model_config.max_position_embeddings,
        cache_dir=training_config.cache_dir
    )

    val_dataset = None
    if val_files:
        val_dataset = ChatDataset(
            val_files,
            tokenizer,
            max_length=model_config.max_position_embeddings,
            cache_dir=training_config.cache_dir
        )
        print(f"✓ Validation samples: {len(val_dataset)}")

    # 5. Model Initialization
    print("\n🤖 Initializing model...")
    model = ChatModel(model_config)
    total_params, trainable_params = model.count_parameters()
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    print(f"✓ Model size: ~{total_params * 4 / 1e9:.2f}GB (fp32)")

    # 6. Training
    print("\n🚀 Initializing trainer...")
    trainer = TrainingService(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        config=training_config
    )

    print("\n" + "="*60)
    print("  STARTING TRAINING")
    print("="*60 + "\n")
    trainer.train()

    # 7. Post-Training Generation Test
    print("\n" + "="*60)
    print("  TESTING GENERATION")
    print("="*60 + "\n")

    generation_service = GenerationService(model, tokenizer)
    test_prompts = [
        "User: What is deep learning?",
        "User: How do I write a Python function?",
    ]

    for prompt in test_prompts:
        print(f"\n💬 Prompt: {prompt}")
        response = get_response(generation_service, prompt)
        print(f"🤖 Response: {response}\n")

    print("="*60)
    print("  ✅ TRAINING COMPLETED SUCCESSFULLY")
    print(f"  📁 Final model checkpoints in: {training_config.output_dir}")
    print("="*60 + "\n")

    # 8. Interactive Session
    print("\n" + "="*60)
    print("  INTERACTIVE CHAT SESSION")
    print("  (Type 'quit' or 'exit' to end)")
    print("="*60 + "\n")

    while True:
        user_prompt = input("You: ")
        if user_prompt.lower() in ["quit", "exit"]:
            print("\n👋 Exiting interactive session. Goodbye!")
            break

        if not user_prompt.strip().lower().startswith("user:"):
             user_prompt = f"User: {user_prompt}"

        response = get_response(generation_service, user_prompt)
        print(f"🤖 Bot: {response}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ A fatal error occurred: {e}")
        import traceback
        traceback.print_exc()