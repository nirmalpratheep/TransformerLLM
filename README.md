# TransformerLLM BPE Tokenizer

A high-performance, byte-level Byte Pair Encoding (BPE) tokenizer implementation designed for Transformer-based LLMs. This project provides a robust wrapper around the HuggingFace `tokenizers` backend with specialized training and inference pipelines.

## Technical Architecture

### Core Components
The system consists of three primary layers:
1.  **Inference Layer (`bpeTokenizer/tokenizer.py`)**: A high-level wrapper that manages model states, handles special tokens (e.g., `<|endoftext|>`), and provides automatic transition between standard and `ByteLevel` tokenization modes.
2.  **Training Pipeline (`bpeTokenizer/train_tokenizer.py`)**: An optimized BPE training script that enforces specific vocabulary ordering: Special Tokens → Base Bytes (0-255) → Learned Merges.
3.  **Interface Layer (`bpeTokenizer/adapters.py`)**: Provides a decoupled entry point for test adapters and integration.

## Key Features

- **Byte-Level BPE**: Uses the GPT-style byte-to-character mapping to ensure 100% coverage of the UTF-8 space without unknown tokens.
- **Special Token Management**: Configurable special tokens injected at the start of the vocabulary indices. Specifically optimized for document boundary markers like `<|endoftext|>`.
- **Memory Efficient**: Supports `encode_iterable` for lazy tokenization of massive datasets without saturating host memory.
- **Dynamic Configuration**: Training parameters (vocab size, frequency thresholds, special tokens) are managed via YAML schemas.

## Getting Started

### Installation
Ensure dependencies are managed via `uv`:
```powershell
uv add tokenizers pyyaml
```

### Training a New Tokenizer
Configure your parameters in `config/tokenizer_config.yaml`:
```yaml
vocab_size: 5000
min_frequency: 2
special_tokens: ["<|endoftext|>"]
files: ["data/TinyStoriesV2-GPT4-valid.txt"]
```
Run the training script:
```powershell
uv run python bpeTokenizer/train_tokenizer.py --config config/tokenizer_config.yaml
```
Output binaries (`vocab.json`, `merges.txt`) are stored in the `./tokenizer` directory.

### Basic Usage
```python
from bpeTokenizer import Tokenizer

# Load trained model
tokenizer = Tokenizer.from_files(
    "tokenizer/vocab.json", 
    "tokenizer/merges.txt", 
    special_tokens=["<|endoftext|>"]
)

# Encoding
ids = tokenizer.encode("Hello World!")

# Decoding
text = tokenizer.decode(ids)
```

## Verification
Run the verification module from the project root:
```powershell
uv run python -m bpeTokenizer.verify_tokenizer
```
This suite demonstrates base character merging, special token handling, and model loading from disk.

## Implementation Details

- **Initial Alphabet**: The trainer is forced to include `pre_tokenizers.ByteLevel.alphabet()` to ensure the base vocabulary contains all single-byte representations.
- **Decoding Strategy**: The `decode` method intelligently switches between manual string joining (for simple models) and HF's `ByteLevel` decoder (for trained models) to prevent artifacts such as unintended spaces.
- **Special Token Handling**: Document splitting is performed on raw text using special tokens before BPE merges are applied to ensure strict delimiter boundaries.
