"""
Maximized Character-Byte Fallback Tokenizer.
"""
import json
import os

class CharByteFallbackTokenizer:
    def __init__(self, vocab_file=None):
        self.byte_fallback_len = 256
        self.vocab_file = vocab_file or os.path.join(os.path.dirname(__file__), "vocab.json")
        
        if os.path.exists(self.vocab_file):
            with open(self.vocab_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.char_to_id = data["char_to_id"]
            self.id_to_char = {int(k): v for k, v in data["id_to_char"].items()}
        else:
            self.char_to_id = {}
            self.id_to_char = {}
            
        self.vocab_size = self.byte_fallback_len + len(self.char_to_id)

    def train_on_corpus(self, text, max_chars=900): # Expanded to 900
        from collections import Counter
        counts = Counter(text)
        common_chars = [ch for ch, _ in counts.most_common(max_chars)]
        
        self.char_to_id = {ch: i + self.byte_fallback_len for i, ch in enumerate(common_chars)}
        self.id_to_char = {i + self.byte_fallback_len: ch for i, ch in enumerate(common_chars)}
        self.vocab_size = self.byte_fallback_len + len(self.char_to_id)
        
        with open(self.vocab_file, "w", encoding="utf-8") as f:
            json.dump({"char_to_id": self.char_to_id, "id_to_char": self.id_to_char}, f)

    def encode(self, text):
        ids = []
        for ch in text:
            if ch in self.char_to_id:
                ids.append(self.char_to_id[ch])
            else:
                for b in ch.encode("utf-8"):
                    ids.append(b)
        return ids

    def decode(self, ids):
        result = []
        byte_buffer = bytearray()
        
        for idx in ids:
            if idx < self.byte_fallback_len:
                byte_buffer.append(idx)
            else:
                if byte_buffer:
                    result.append(byte_buffer.decode("utf-8", errors="replace"))
                    byte_buffer.clear()
                result.append(self.id_to_char[idx])
                
        if byte_buffer:
            result.append(byte_buffer.decode("utf-8", errors="replace"))
            
        return "".join(result)

def load():
    tokenizer = CharByteFallbackTokenizer()
    if tokenizer.vocab_size == 256:
        potential_data_paths = ["../data/train_corpus.txt", "data/train_corpus.txt"]
        for path in potential_data_paths:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                tokenizer.train_on_corpus(text)
                break
    return tokenizer