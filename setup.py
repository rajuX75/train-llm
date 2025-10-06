"""
GOOGLE COLAB INSTALLATION & SETUP SCRIPT
Run this in the first cell of your Colab notebook
"""

# ============== CELL 1: Check GPU and Install Dependencies ==============
print("="*70)
print("  CHAT MODEL TRAINING - GOOGLE COLAB SETUP")
print("="*70)

# Check GPU availability
print("\n[1/6] Checking GPU...")
import subprocess
try:
    gpu_info = subprocess.check_output(['nvidia-smi'], text=True)
    print("✓ GPU detected:")
    print(gpu_info.split('\n')[8])  # Print GPU name line
except:
    print("⚠️  No GPU detected! Training will be very slow on CPU.")
    print("   Go to Runtime > Change runtime type > Hardware accelerator > GPU")

# Install PyTorch with CUDA support
print("\n[2/6] Installing PyTorch with CUDA 11.8...")
import sys
!{sys.executable} -m pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
print("\n[3/6] Installing core dependencies...")
!{sys.executable} -m pip install -q transformers datasets accelerate sentencepiece
!{sys.executable} -m pip install -q einops rotary-embedding-torch
!{sys.executable} -m pip install -q tqdm numpy

# Install optional but recommended packages
print("\n[4/6] Installing optional packages...")
!{sys.executable} -m pip install -q wandb  # For experiment tracking
!{sys.executable} -m pip install -q bitsandbytes  # For 8-bit optimization

# Try to install Flash Attention (may fail on some GPUs)
print("\n[5/6] Installing Flash Attention (optional)...")
try:
    !{sys.executable} -m pip install -q flash-attn --no-build-isolation
    print("✓ Flash Attention installed successfully")
except:
    print("⚠️  Flash Attention installation failed (not critical)")
    print("   Training will use standard attention instead")

# Verify installations
print("\n[6/6] Verifying installations...")
import torch
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

import transformers
import sentencepiece
import einops
print(f"✓ Transformers: {transformers.__version__}")
print(f"✓ SentencePiece: {sentencepiece.__version__}")

print("\n" + "="*70)
print("  ✓ INSTALLATION COMPLETE!")
print("="*70)

# ============== CELL 2: Mount Google Drive (Optional) ==============
"""
# Uncomment and run this cell if you want to save models to Google Drive
print("\n[Optional] Mounting Google Drive...")
from google.colab import drive
drive.mount('/content/drive')
print("✓ Google Drive mounted at /content/drive")

# Set paths to use Google Drive
DATA_DIR = "/content/drive/MyDrive/chat_model/data"
OUTPUT_DIR = "/content/drive/MyDrive/chat_model/models"
CACHE_DIR = "/content/drive/MyDrive/chat_model/cache"

import os
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

print(f"✓ Directories created in Google Drive")
print(f"  Data: {DATA_DIR}")
print(f"  Models: {OUTPUT_DIR}")
print(f"  Cache: {CACHE_DIR}")
"""

# ============== CELL 3: Create Directories ==============
print("\n[Setup] Creating working directories...")
import os

# Use local paths (faster but lost when runtime disconnects)
DATA_DIR = "./data"
OUTPUT_DIR = "./models"
CACHE_DIR = "./cache"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

print("✓ Directories created:")
print(f"  Data: {DATA_DIR}")
print(f"  Models: {OUTPUT_DIR}")
print(f"  Cache: {CACHE_DIR}")

# ============== CELL 4: Upload Data (Optional) ==============
"""
# Option 1: Upload files directly to Colab
print("\n[Data] Upload your training data files...")
from google.colab import files
uploaded = files.upload()

import shutil
for filename in uploaded.keys():
    shutil.move(filename, f"{DATA_DIR}/{filename}")
    print(f"✓ Moved {filename} to {DATA_DIR}")
"""

"""
# Option 2: Copy from Google Drive (if mounted)
print("\n[Data] Copying data from Google Drive...")
# Update this path to where your data is stored in Drive
DRIVE_DATA_PATH = "/content/drive/MyDrive/your_data_folder"
!cp {DRIVE_DATA_PATH}/*.txt {DATA_DIR}/
print(f"✓ Copied data files to {DATA_DIR}")
"""

"""
# Option 3: Download from URL
print("\n[Data] Downloading data from URL...")
import urllib.request
urls = [
    "https://example.com/conversation1.txt",
    "https://example.com/conversation2.txt",
]
for i, url in enumerate(urls):
    urllib.request.urlretrieve(url, f"{DATA_DIR}/conversation_{i+1}.txt")
    print(f"✓ Downloaded conversation_{i+1}.txt")
"""

# ============== CELL 5: Quick Test ==============
print("\n[Test] Running quick installation test...")

# Test PyTorch
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test tensor creation
x = torch.randn(2, 3)
if torch.cuda.is_available():
    x = x.cuda()
    print(f"✓ GPU tensor created: {x.device}")
else:
    print("⚠️  Running on CPU")

# Test SentencePiece
import sentencepiece as spm
print(f"✓ SentencePiece: {spm.__version__}")

print("\n✓ All tests passed! Ready to train.")

# ============== CELL 6: Clone/Upload Training Script ==============
"""
# Option 1: Upload the training script
print("\n[Script] Upload the training script...")
from google.colab import files
uploaded = files.upload()
print("✓ Training script uploaded")
"""

"""
# Option 2: Clone from GitHub
print("\n[Script] Cloning from GitHub...")
!git clone https://github.com/yourusername/chat-model-training.git
%cd chat-model-training
print("✓ Repository cloned")
"""

"""
# Option 3: Download from URL
print("\n[Script] Downloading training script...")
!wget https://example.com/train_chat_model.py -O train_chat_model.py
print("✓ Training script downloaded")
"""

# ============== CELL 7: Setup Wandb (Optional) ==============
"""
# If you want to use Weights & Biases for experiment tracking
print("\n[Optional] Setting up Wandb...")
import wandb

# Login to wandb (you'll need an API key from https://wandb.ai)
wandb.login()

# Or set API key directly
# wandb.login(key="your_api_key_here")

print("✓ Wandb configured")
"""

# ============== CELL 8: Quick Memory Check ==============
print("\n[Memory] Checking available resources...")

# Check RAM
import psutil
ram_gb = psutil.virtual_memory().total / 1e9
ram_available_gb = psutil.virtual_memory().available / 1e9
print(f"✓ Total RAM: {ram_gb:.1f} GB")
print(f"✓ Available RAM: {ram_available_gb:.1f} GB")

# Check GPU memory
if torch.cuda.is_available():
    gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    gpu_mem_reserved = torch.cuda.memory_reserved(0) / 1e9
    gpu_mem_allocated = torch.cuda.memory_allocated(0) / 1e9
    print(f"✓ GPU Memory Total: {gpu_mem_total:.2f} GB")
    print(f"✓ GPU Memory Reserved: {gpu_mem_reserved:.2f} GB")
    print(f"✓ GPU Memory Allocated: {gpu_mem_allocated:.2f} GB")
    print(f"✓ GPU Memory Free: {gpu_mem_total - gpu_mem_reserved:.2f} GB")

# Check disk space
disk = psutil.disk_usage('/')
disk_free_gb = disk.free / 1e9
print(f"✓ Free Disk Space: {disk_free_gb:.1f} GB")

# Recommendations
print("\n[Recommendations]")
if ram_available_gb < 10:
    print("⚠️  Low RAM. Consider reducing batch size.")
if torch.cuda.is_available() and gpu_mem_total < 12:
    print("⚠️  Limited GPU memory. Use smaller model or gradient checkpointing.")
if disk_free_gb < 5:
    print("⚠️  Low disk space. May not be enough for checkpoints.")

# ============== CELL 9: Configuration Presets ==============
print("\n[Config] Recommended configurations for different GPUs:\n")

configs = {
    "T4 (16GB)": {
        "batch_size": 2,
        "gradient_accumulation_steps": 16,
        "hidden_size": 1024,
        "num_hidden_layers": 12,
        "max_position_embeddings": 2048,
    },
    "A100 (40GB)": {
        "batch_size": 8,
        "gradient_accumulation_steps": 4,
        "hidden_size": 2048,
        "num_hidden_layers": 24,
        "max_position_embeddings": 4096,
    },
    "V100 (16GB)": {
        "batch_size": 4,
        "gradient_accumulation_steps": 8,
        "hidden_size": 1536,
        "num_hidden_layers": 16,
        "max_position_embeddings": 2048,
    },
    "Free Tier (12GB)": {
        "batch_size": 1,
        "gradient_accumulation_steps": 32,
        "hidden_size": 768,
        "num_hidden_layers": 8,
        "max_position_embeddings": 1024,
    }
}

for gpu_type, config in configs.items():
    print(f"{gpu_type}:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

print("Copy and modify these in your TrainingConfig!")

# ============== FINAL MESSAGE ==============
print("="*70)
print("  🚀 SETUP COMPLETE - READY TO TRAIN!")
print("="*70)
print("\nNext steps:")
print("1. Upload or create your training data in the data directory")
print("2. Copy the optimized training script into a new cell")
print("3. Modify the configuration if needed")
print("4. Run the training script")
print("\nFor long training sessions:")
print("- Consider mounting Google Drive to save checkpoints")
print("- Enable Colab Pro for longer runtime")
print("- Use wandb for remote monitoring")
print("="*70)
