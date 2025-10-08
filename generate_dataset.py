import os
from datasets import load_dataset

def format_conversation(prompt, response):
    """Formats a conversation into the required 'User: ...\nAssistant: ...' format."""
    return f"User: {prompt}\nAssistant: {response}\n\n"

def generate_dataset_from_hub(
    filename="data/openassistant_conversations.txt",
    dataset_name="OpenAssistant/oasst1",
    split="train",
):
    """
    Downloads a conversational dataset from the Hugging Face Hub, processes it,
    and saves it to a text file.

    This function processes message trees to extract two-turn conversations
    (a user prompt followed by an assistant's response).

    Args:
        filename (str): The path to the output file.
        dataset_name (str): The name of the dataset on the Hugging Face Hub.
        split (str): The dataset split to use (e.g., 'train', 'validation').
    """
    print(f"📥 Downloading dataset '{dataset_name}'...")
    dataset = load_dataset(dataset_name, split=split)
    print("✅ Dataset downloaded successfully.")

    # Ensure the output directory exists
    output_dir = os.path.dirname(filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processed_conversations = 0
    with open(filename, "w", encoding="utf-8") as f:
        # The dataset is a flat list of messages that need to be reconstructed into trees.
        # We can group messages by their message_tree_id.
        message_trees = {}
        for msg in dataset:
            if msg["message_tree_id"] not in message_trees:
                message_trees[msg["message_tree_id"]] = []
            message_trees[msg["message_tree_id"]].append(msg)

        # Process each conversation tree
        for tree_id, messages in message_trees.items():
            # Create a dictionary of messages by their ID for easy lookup
            msg_map = {msg["message_id"]: msg for msg in messages}

            # Find root messages (prompts)
            for msg in messages:
                if msg["parent_id"] is None and msg["role"] == "prompter":
                    # Find direct assistant replies to this prompt
                    for reply in messages:
                        if reply["parent_id"] == msg["message_id"] and reply["role"] == "assistant":
                            if msg["lang"] == "en" and reply["lang"] == "en":
                                formatted_conv = format_conversation(msg["text"], reply["text"])
                                f.write(formatted_conv)
                                processed_conversations += 1

    print(f"✅ Successfully processed and saved {processed_conversations} conversations to '{filename}'.")

if __name__ == "__main__":
    generate_dataset_from_hub()