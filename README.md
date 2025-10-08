# Modular Chat Model Training

This project provides a modular and optimized pipeline for training a chat model from scratch. It is designed with a Domain-Driven Design (DDD) and modular monolith architecture, making it easy to understand, maintain, and extend.

## 🚀 Features

- **Modular Architecture:** The codebase is organized into distinct domains (e.g., `data_management`, `tokenization`, `model_core`, `training`), promoting separation of concerns and scalability.
- **Customizable Model:** The model architecture is highly configurable, allowing you to experiment with different hyperparameters.
- **Efficient Data Handling:** The `ChatDataset` class uses lazy loading to efficiently handle large datasets.
- **Reproducibility:** The project includes seeding to ensure reproducible training runs.

## 🏛️ Architecture

The project follows a modular monolith architecture inspired by Domain-Driven Design (DDD) and Hexagonal Architecture. The core logic is decoupled from external frameworks and technologies, making the system more flexible and testable.

The main components are:

- **`src/app`**: The application layer, responsible for orchestrating the different domains to execute the training pipeline.
- **`src/domains`**: Contains the core business logic, organized into sub-domains:
  - **`data_management`**: Handles dataset creation and loading.
  - **`tokenization`**: Manages the tokenizer, including training and encoding.
  - **`model_core`**: Defines the neural network architecture for the chat model.
  - **`training`**: Contains the training loop and related services.
  - **`inference`**: Provides services for generating text with a trained model.
- **`src/shared`**: Includes shared utilities and configuration that are used across different domains.

## 🔧 Setup and Installation

Follow these steps to set up the project on your local machine.

### 1. Clone the Repository

```bash
git clone <repository_url>
cd <repository_directory>
```

### 2. Install Dependencies

The required Python packages are listed in `requirements.txt`. You can install them using pip:

```bash
pip install -r requirements.txt
```

### 3. Install the Project

To make the local `src` package available to your Python environment, install the project in editable mode:

```bash
pip install -e .
```

## ▶️ How to Run

### 1. Generate the Dataset

This project includes a script to download and process a high-quality conversational dataset from the Hugging Face Hub. To generate the dataset, run the following command from the root of the repository:

```bash
python generate_dataset.py
```

This will download the `OpenAssistant/oasst1` dataset, format it into the required `User: ...\nAssistant: ...` structure, and save it to `data/openassistant_conversations.txt`.

The generated data directory is included in the `.gitignore` file, so the dataset will not be committed to your repository.

### 2. Start the Training

To start the training process, run the main application script:

```bash
python -m src.app.main
```

The script will:
1.  Initialize the tokenizer (and train it if a model doesn't exist).
2.  Create the training and validation datasets.
3.  Initialize the chat model.
4.  Start the training process, saving checkpoints to the `./models` directory.
5.  Run a generation test after training is complete.

## ⚙️ Configuration

The project's configuration is managed in `src/shared/config/config.yaml`. You can modify this file to adjust hyperparameters for the model and training process.

The configuration is split into two main sections:
- **`model`**: Defines the model's architecture (e.g., `hidden_size`, `num_hidden_layers`, `vocab_size`).
- **`training`**: Defines the training parameters (e.g., `batch_size`, `learning_rate`, `num_epochs`).

Enjoy training your own chat model!