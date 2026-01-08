
from .tokenizer import Tokenizer
import os
import json

def test_tokenizer_simple():
    # Setup simple vocab and merges
    # text: "hello world"
    # vocab needs: h, e, l, o, w, r, d, space
    # let's try a smaller example: "ab ac"
    # a, b, c, space
    
    vocab = {
        0: b'a',
        1: b'b',
        2: b'c',
        3: b' ',
        4: b'ab', # merge 1
        5: b'ac', # merge 2
    }
    
    merges = [
        (b'a', b'b'),
        (b'a', b'c'),
    ]
    
    t = Tokenizer(vocab, merges)
    
    # Test encode "ab ac"
    # "ab" -> 4
    # " " -> 3
    # "ac" -> 5
    ids = t.encode("ab ac")
    print(f"Encode 'ab ac': {ids}")
    assert ids == [4, 3, 5]
    
    # Test decode
    decoded = t.decode(ids)
    print(f"Decode {ids}: '{decoded}'")
    assert decoded == "ab ac"

    print("Simple test passed!")

def test_special_tokens():
    vocab = {
        0: b'a',
        1: b'b',
    }
    merges = [] # No merges
    special = ["<|endoftext|>"]
    
    t = Tokenizer(vocab, merges, special_tokens=special)
    
    # Check special token ID
    ids = t.encode("<|endoftext|>")
    assert len(ids) == 1
    sid = ids[0]
    print(f"Special token ID: {sid}")
    
    text = "a<|endoftext|>b"
    ids = t.encode(text)
    print(f"Encode '{text}': {ids}")
    # 0, sid, 1
    assert ids == [0, sid, 1]
    
    decoded = t.decode(ids)
    print(f"Decode {ids}: '{decoded}'")
    assert decoded == text

    print("Special token test passed!")

def test_missing_chars():
    # If a char is not in vocab, current implementation might error or skip
    # Let's see what happens.
    # Current code: "ids.append(self.vocab_inv[b])" -> KeyError
    pass

def test_from_files():
    # Adjust paths for where the tokenizer outputs are located
    vocab_path = os.path.join("tokenizer", "vocab.json")
    merges_path = os.path.join("tokenizer", "merges.txt")
    
    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        print(f"Skipping from_files test: {vocab_path} or {merges_path} not found.")
        return

    print("Testing from_files...")
    # Load from the trained files
    t = Tokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])
    
    text = "Hello world<|endoftext|>"
    ids = t.encode(text)
    print(f"Encoded '{text}': {ids}")
    decoded = t.decode(ids)
    print(f"Decoded: '{decoded}'")
    assert decoded == text
    print("from_files test passed!")

if __name__ == "__main__":
    test_tokenizer_simple()
    test_special_tokens()
    test_from_files()
