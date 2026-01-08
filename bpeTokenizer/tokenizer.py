
import json
from typing import Dict, List, Tuple, Iterable, Iterator, Optional
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: Optional[List[str]] = None):
        self.special_tokens = special_tokens if special_tokens is not None else []
        
        # Convert vocab to string -> int
        # HF BPE expects {token: id}
        vocab_hf = {}
        for idx, token_bytes in vocab.items():
            try:
                # Try UTF-8
                token_str = token_bytes.decode('utf-8')
            except UnicodeDecodeError:
                # Fallback: using repr or latin-1? 
                token_str = token_bytes.decode('utf-8', errors='replace')
            vocab_hf[token_str] = idx

        # Convert merges to list of (str, str)
        merges_hf = []
        for p1, p2 in merges:
            p1_str = p1.decode('utf-8', errors='replace')
            p2_str = p2.decode('utf-8', errors='replace')
            merges_hf.append((p1_str, p2_str))

        # Initialize BPE model
        model = BPE(vocab_hf, merges_hf)
        
        # Initialize Tokenizer
        self.tokenizer = HFTokenizer(model)
        
        # Check if vocab uses ByteLevel characters (e.g. Ġ)
        # Ġ is \u0120
        self.use_byte_level = any('\u0120' in t for t in vocab_hf.keys())
        
        if self.use_byte_level:
            self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
            self.tokenizer.decoder = ByteLevelDecoder()
        
        # Handling special tokens
        if self.special_tokens:
             self.tokenizer.add_special_tokens(self.special_tokens)

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[List[str]] = None):
        import json
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            raw_vocab = json.load(f)
            vocab = {}
            # Check if it's token->id or id->token
            first_key = next(iter(raw_vocab)) if raw_vocab else None
            is_token_to_id = False
            if first_key is not None:
                if isinstance(raw_vocab[first_key], int):
                    is_token_to_id = True
            
            if is_token_to_id:
                for token_str, token_id in raw_vocab.items():
                     vocab[token_id] = token_str.encode('utf-8')
            else:
                for k, v in raw_vocab.items():
                    if isinstance(v, str):
                        vocab[int(k)] = v.encode('utf-8')
                    else:
                        vocab[int(k)] = bytes(v)

        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip() or line.startswith("#"): continue
                parts = line.strip().split()
                if len(parts) == 2:
                    merges.append((parts[0].encode('utf-8'), parts[1].encode('utf-8')))
        
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> List[int]:
        # encode returns Encoding object
        return self.tokenizer.encode(text).ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: List[int]) -> str:
        if self.use_byte_level:
             # Use HF decode logic which handles ByteLevel reversing
             return self.tokenizer.decode(ids, skip_special_tokens=False)
        else:
             # Use simple concatenation to avoid HF's default space insertion
             # This mirrors the simple BPE behavior requested
             tokens = []
             for i in ids:
                 t = self.tokenizer.id_to_token(i)
                 if t is not None:
                     tokens.append(t)
             return "".join(tokens)
