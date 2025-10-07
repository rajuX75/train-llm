import random

def generate_large_dataset(filename="data/large_conversations.txt", num_samples=1000):
    """
    Generates a large conversational dataset with diverse topics.

    Args:
        filename (str): The path to the output file.
        num_samples (int): The number of user-assistant conversation pairs to generate.
    """
    topics = {
        "science": [
            ("What is the powerhouse of the cell?", "The mitochondrion is known as the powerhouse of the cell."),
            ("What is the speed of light?", "The speed of light in a vacuum is approximately 299,792 kilometers per second."),
            ("What are the primary colors?", "The primary colors in light are red, green, and blue (RGB)."),
            ("What is photosynthesis?", "Photosynthesis is the process used by plants, algae, and certain bacteria to convert light energy into chemical energy."),
        ],
        "history": [
            ("Who was the first president of the United States?", "George Washington was the first president of the United States."),
            ("When did World War II end?", "World War II ended in 1945."),
            ("What was the Renaissance?", "The Renaissance was a period in European history marking the transition from the Middle Ages to modernity."),
        ],
        "technology": [
            ("What is a neural network?", "A neural network is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates."),
            ("What is the difference between RAM and ROM?", "RAM (Random Access Memory) is volatile memory that stores data temporarily, while ROM (Read-Only Memory) is non-volatile memory that stores data permanently."),
            ("What is object-oriented programming?", "Object-oriented programming (OOP) is a programming paradigm based on the concept of 'objects', which can contain data and code."),
        ],
        "creative": [
            ("Tell me a joke.", "Why don't scientists trust atoms? Because they make up everything!"),
            ("Write a short poem about nature.", "The wind whispers through the trees,\nA gentle, rustling, soft-breeze.\nThe sun warms the fertile ground,\nWhere life and beauty can be found."),
            ("What is a good book to read?", "It depends on your taste, but 'Dune' by Frank Herbert is a classic science fiction novel that many enjoy."),
        ],
    }

    all_conversations = []
    for category in topics.values():
        all_conversations.extend(category)

    with open(filename, "w", encoding="utf-8") as f:
        for _ in range(num_samples):
            user_q, assistant_a = random.choice(all_conversations)
            f.write(f"User: {user_q}\n")
            f.write(f"Assistant: {assistant_a}\n\n")

    print(f"✅ Successfully generated {num_samples} samples in '{filename}'")

if __name__ == "__main__":
    generate_large_dataset()