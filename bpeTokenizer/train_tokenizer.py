
import argparse
import yaml
import os
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from typing import List, Iterator

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def iter_docs(file_path: str, delimiter: str = "<|endoftext|>") -> Iterator[str]:
    # Read file line by line to support large files
    # However, if delimiter is <|endoftext|>, it might not be line-based.
    # But usually huge datasets are line based or split by that token.
    # For safety with potentially huge files, we can read chunks.
    # But for BPE training, we need reachable chunks. 
    # Let's read the whole file if it fits in memory (easy) or stream it.
    # Given the constraint: "split the text based on <|endoftext|>"
    # I'll implement a generator.
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        for doc in text.split(delimiter):
            if doc.strip():
                yield doc

def train_tokenizer(config_path: str, input_file: str = None):
    config = load_config(config_path)
    
    files = config.get('files', [])
    if input_file:
        files = [input_file]
        
    if not files:
        raise ValueError("No input files specified in config or arguments.")
        
    vocab_size = config.get('vocab_size', 5000)
    min_frequency = config.get('min_frequency', 2)
    special_tokens = config.get('special_tokens', [])
    
    # Initialize tokenizer
    # We use BPE model
    tokenizer = Tokenizer(models.BPE())
    
    # Pre-tokenizer: ByteLevel is standard for GPT-2 style
    # But the Prompt assumes just "text".
    # I will use ByteLevel as it's robust.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    # Define initial alphabet: bytes 0-255
    # tokenizers expects list of strings
    # We use byte level so we want the characters corresponding to bytes 0-255
    # ByteLevel pre-tokenizer maps bytes to chars.
    # Actually, BpeTrainer with ByteLevel PreTokenizer should find them.
    # BUT user insisted on "add byte 0 to 255 (ascii) and merge of these bytes"
    # To force this, we can set initial_alphabet.
    # Be careful: ByteLevel maps bytes to specific unicode chars (GPT-2 style).
    # We should use the same mapping or just use PreTokenizer's alphabet.
    # BpeTrainer has `initial_alphabet` param.
    # If we use pre_tokenizers.ByteLevel.alphabet():
    initial_alphabet = pre_tokenizers.ByteLevel.alphabet()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=initial_alphabet
    )
    
    # Iterator
    def data_iterator():
        for file_path in files:
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} not found.")
                continue
            print(f"Processing {file_path}...")
            yield from iter_docs(file_path)

    print("Training tokenizer...")
    tokenizer.train_from_iterator(data_iterator(), trainer=trainer)
    
    # Save
    print("Saving tokenizer...")
    output_dir = "."
    tokenizer.model.save(output_dir, "")
    
    # Rename files to remove leading hyphen if present
    if os.path.exists(os.path.join(output_dir, "-vocab.json")):
        os.replace(os.path.join(output_dir, "-vocab.json"), os.path.join(output_dir, "vocab.json"))
    if os.path.exists(os.path.join(output_dir, "-merges.txt")):
        os.replace(os.path.join(output_dir, "-merges.txt"), os.path.join(output_dir, "merges.txt"))
 
    print(f"Tokenizer saved to {output_dir}/vocab.json and {output_dir}/merges.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Updated default config path relative to script location or root
    # User said "config yaml inside tokenizer_config.yaml" which implies inside config folder
    # Assuming script is run from root or bpeTokenizer? Default is usually root.
    # If script moved to bpeTokenizer/, then config is ../config/tokenizer_config.yaml
    parser.add_argument("--config", type=str, default="../config/tokenizer_config.yaml", help="Path to config file")
    parser.add_argument("--input", type=str, default=None, help="Override input file")
    args = parser.parse_args()
    
    train_tokenizer(args.config, args.input)
