"""
Comprehensive dataset generator and tokenizer for local AI Assistant.
Full ASCII character coverage ensures zero unmapped tokens or prompt mangling.
"""
import os
import pickle
import string
import numpy as np

CONVERSATIONS = [
    ("Hi", "Hello! I am your local AI assistant. How can I help you today?"),
    ("hi", "Hello! Glad to see you. What would you like to explore or build?"),
    ("Hello", "Greetings! I am ready to help you with questions, coding, and learning."),
    ("hello", "Hello! How can I assist you right now?"),
    ("Hey", "Hey there! Ready to assist you. What is on your mind?"),
    ("hey", "Hey! How can I assist your coding or research today?"),
    ("Good morning", "Good morning! I hope you have a productive day ahead. What are we working on?"),
    ("Good evening", "Good evening! How can I help you with your projects tonight?"),
    ("How are you?", "I am running smoothly on your local machine and ready to help!"),
    ("how are you", "I am doing well and completely offline on your computer. How can I assist?"),
    ("Who are you?", "I am a local Transformer neural network assistant running 100% on your device without cloud APIs."),
    ("who are you", "I am your private local AI assistant, trained directly with nanoGPT."),
    ("What is your name?", "I am NanoGPT Assistant, your private offline neural network."),
    ("what is your name", "My name is NanoGPT Assistant. I operate locally on your hardware."),
    ("Are you ChatGPT?", "No, I am an independent, local Transformer model running on your laptop with zero external API dependencies."),
    ("Do you need API keys?", "No! I run 100% locally on your computer with complete privacy and zero API keys."),
    ("What is Python?", "Python is a versatile programming language known for readable syntax and powerful machine learning libraries."),
    ("Can you write a Python hello world?", "Sure! In Python: print('Hello, World!')"),
    ("How do I write a function in Python?", "In Python, use def: def greet(name): return f'Hello {name}'"),
    ("Can you write a function to add two numbers?", "Yes: def add(a, b): return a + b"),
    ("What is a list in Python?", "A list is an ordered, mutable sequence written with square brackets, like: items = [1, 2, 3]"),
    ("What is a dictionary in Python?", "A dictionary stores key-value pairs written with curly braces, like: user = {'name': 'Alice', 'role': 'Admin'}"),
    ("What is a loop in Python?", "A loop repeats code over a sequence. For example: for i in range(5): print(i)"),
    ("How do I handle errors in Python?", "Use try-except blocks: try: result = 10 / x except ZeroDivisionError: print('Cannot divide by zero')"),
    ("What is artificial intelligence?", "Artificial intelligence is the field of computer science creating machines capable of reasoning, learning, and problem solving."),
    ("What is machine learning?", "Machine learning enables computers to learn patterns from data automatically without explicit rule-based programming."),
    ("What is deep learning?", "Deep learning is a branch of machine learning utilizing multi-layer neural networks to learn representations from raw data."),
    ("What is a transformer model?", "A transformer is a neural network architecture based on self-attention mechanisms that process sequence tokens in parallel."),
    ("What is self-attention?", "Self-attention computes dynamic weights between all tokens in a sequence, capturing long-range semantic relationships."),
    ("How do neural networks learn?", "They learn through gradient descent and backpropagation, iteratively updating weights to minimize prediction loss."),
    ("What is a token?", "A token is a basic unit of text processed by a language model, such as a character, subword, or word."),
    ("What is GPU?", "A Graphics Processing Unit provides massively parallel compute cores optimized for matrix multiplications in deep learning."),
    ("What is CPU?", "A Central Processing Unit is the primary general-purpose processor executing OS instructions and sequential computing."),
    ("What is RAM?", "Random Access Memory provides fast, temporary volatile storage for active program data and model weights."),
    ("What is 2 + 2?", "2 + 2 equals 4."),
    ("What is 10 * 10?", "10 * 10 equals 100."),
    ("What is gravity?", "Gravity is a fundamental natural force by which physical bodies attract one another with force proportional to their masses."),
    ("Why is the sky blue?", "The sky appears blue because molecules in Earth's atmosphere scatter short blue wavelengths of sunlight more than other colors."),
    ("What is open source?", "Open source software makes its source code freely accessible for anyone to inspect, modify, and enhance."),
    ("What is Git?", "Git is a distributed version control system designed to track changes in source code during software development."),
    ("What is an algorithm?", "An algorithm is a step-by-step procedure of well-defined instructions designed to solve a specific problem."),
    ("Thank you", "You are very welcome! Let me know if you need any other assistance."),
    ("thank you", "Happy to help! Feel free to ask anytime."),
    ("Thanks", "You are welcome! Have fun coding."),
    ("Bye", "Goodbye! Have a fantastic and productive day."),
    ("bye", "Goodbye! I will be here whenever you need me."),
    ("Help me", "I am here to help! Tell me what problem or question you are working on."),
    ("What can you do?", "I can answer questions, explain concepts, help with Python code, and generate local text.")
]

def build_dataset_text():
    # Build cleanly formatted conversation turns
    blocks = []
    for user_msg, asst_msg in CONVERSATIONS:
        blocks.append(f"User: {user_msg}\nAssistant: {asst_msg}\n")
    raw = "\n".join(blocks)
    # Repeat so the model sees multiple passes of rich context
    return raw * 80

def prepare():
    dataset_dir = os.path.dirname(os.path.abspath(__file__))
    input_file_path = os.path.join(dataset_dir, 'input.txt')
    
    dataset_text = build_dataset_text()
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(dataset_text)
        
    print(f"Dataset characters: {len(dataset_text):,}")
    
    # Establish complete character set: standard printable ASCII + newline
    # This guarantees that NO valid user input character is ever missing or mapped to space!
    standard_chars = sorted(list(set(string.printable.replace('\r', '').replace('\x0b', '').replace('\x0c', ''))))
    dataset_chars = sorted(list(set(dataset_text)))
    all_chars = sorted(list(set(standard_chars + dataset_chars)))
    
    vocab_size = len(all_chars)
    print(f"Total vocabulary size: {vocab_size} unique characters")
    print(f"Included characters: {''.join(all_chars)}")
    
    stoi = { ch: i for i, ch in enumerate(all_chars) }
    itos = { i: ch for i, ch in enumerate(all_chars) }
    
    def encode(s: str):
        # Map known characters directly; fallback to nearest printable space
        return [stoi.get(c, stoi.get(' ', 0)) for c in s]
        
    def decode(tokens):
        return ''.join([itos.get(i, '?') for i in tokens])
        
    n = len(dataset_text)
    train_data = dataset_text[:int(n * 0.9)]
    val_data = dataset_text[int(n * 0.9):]
    
    train_ids = np.array(encode(train_data), dtype=np.uint16)
    val_ids = np.array(encode(val_data), dtype=np.uint16)
    
    print(f"Train tokens: {len(train_ids):,}")
    print(f"Val tokens: {len(val_ids):,}")
    
    train_ids.tofile(os.path.join(dataset_dir, 'train.bin'))
    val_ids.tofile(os.path.join(dataset_dir, 'val.bin'))
    
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(dataset_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
        
    print("Dataset preparation complete! All ASCII characters mapped successfully.")

if __name__ == '__main__':
    prepare()
