from pathlib import Path

def create_sample_data(data_dir: str = "./data"):
    """Create comprehensive sample training data files."""
    Path(data_dir).mkdir(exist_ok=True)

    sample_conversations = [
        """User: Hello, how are you today?
Assistant: I'm doing well, thank you for asking! I'm here and ready to help you with any questions or tasks you might have. How can I assist you today?

User: Can you explain what machine learning is?
Assistant: Machine learning is a branch of artificial intelligence where computer systems learn and improve from experience without being explicitly programmed. Instead of following rigid rules, ML algorithms identify patterns in data and make decisions based on those patterns.""",

        """User: I need help writing Python code
Assistant: I'd be happy to help you with Python code! Could you tell me more about what you're trying to accomplish?

User: I want to read a CSV file and analyze the data
Assistant: I'll help you read and analyze CSV data using pandas. First, install pandas if you haven't already. Then you can use pd.read_csv() to load your data and various methods like describe(), head(), and info() to explore it.""",

        """User: What's the difference between AI and machine learning?
Assistant: AI is the broader field aimed at creating machines that can perform tasks requiring human-like intelligence. Machine Learning is a specific subset of AI that focuses on algorithms that can learn from experience without being explicitly programmed for every scenario.""",
    ]

    for i, conv in enumerate(sample_conversations):
        file_path = Path(data_dir) / f"conversation_{i+1}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(conv)

    print(f"✓ Created {len(sample_conversations)} sample files in {data_dir}")
    return [str(file_path) for file_path in Path(data_dir).glob("*.txt")]

if __name__ == "__main__":
    create_sample_data()