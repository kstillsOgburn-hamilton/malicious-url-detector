"""
@ Description: retrieves all the url data, balances it among 4 classes, and stores it into a dataset file
@ Author: Kenisha Stills
@ Create Time: 11/10/25
"""

import torch
from typing import List, Dict

class CharTokenizer:
    """
    Character-level tokenizer for URL classification.
    
    Tokenizes URLs character-by-character with special tokens:
    - <pad>: Padding token (index 0)
    - <unk>: Unknown characters (index 1)
    - <cls>: Classification token prepended to sequence (index 2)
    
    Attributes:
        char2id: Dictionary mapping characters to indices
        id2char: Dictionary mapping indices to characters
        pad_id: Index of padding token
        unk_id: Index of unknown token
        cls_id: Index of classification token
    """
    
    def __init__(self):
        # Special tokens
        self.char2id: Dict[str, int] = {"<pad>": 0, "<unk>": 1, "<cls>": 2}
        self.id2char: Dict[int, str] = {0: "<pad>", 1: "<unk>", 2: "<cls>"}

        self.pad_id: int = 0
        self.unk_id: int = 1
        self.cls_id: int = 2

    def build_vocab(self, urls: List[str]) -> None:
        """
        Build vocabulary from list of URLs.
        
        Args:
            urls: List of URL strings to build vocabulary from
        """
        if not urls:
            raise ValueError("Cannot build vocabulary from empty URL list")
            
        for url in urls:
            if not isinstance(url, str):
                continue  # Skip non-string entries
            for ch in url:
                if ch not in self.char2id:
                    idx = len(self.char2id)
                    self.char2id[ch] = idx
                    self.id2char[idx] = ch

    def encode(self, url: str) -> List[int]:
        """
        Encode URL string to list of token IDs.
        
        Args:
            url: URL string to encode
            
        Returns:
            List of token IDs with <cls> prepended
            
        Raises:
            ValueError: If url is not a string
        """
        if not isinstance(url, str):
            raise ValueError(f"Expected string, got {type(url)}")
        
        if not url.strip():
            return [self.cls_id]  # Empty URL -> just CLS token
        
        # Normalize URL for consistency
        url = url.lower().strip()
        
        encoded = [
            self.char2id.get(ch, self.unk_id)
            for ch in url
        ]
        return [self.cls_id] + encoded

    def decode(self, ids: List[int]) -> str:
        """
        Decode list of token IDs to URL string.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded URL string
        """
        return "".join(
            self.id2char.get(i, "<unk>")
            for i in ids
        )

    @property
    def vocab_size(self) -> int:
        """Return size of vocabulary."""
        return len(self.char2id)


import re

class WordTokenizer:
    def __init__(self):
        # Special tokens
        self.word2id = {"<pad>": 0, "<unk>": 1, "<cls>": 2}
        self.id2word = {0: "<pad>", 1: "<unk>", 2: "<cls>"}

        self.pad_id = 0
        self.unk_id = 1
        self.cls_id = 2

        # URL-aware regex designed for your cleaned dataset
        self.pattern = re.compile(
            r"""
            (%[0-9A-Fa-f]{2})           |  # percent-encoding like %2F
            ([A-Za-z0-9]+)              |  # alphanumeric segments
            ([./\-_=&?:@#])             |  # URL delimiters
            (.)                         |  # everything else
            """,
            re.VERBOSE,
        )

    def tokenize(self, url):
        """Tokenize a URL string into regex-derived pieces."""
        if not isinstance(url, str):
            return []
        url = url.lower()
        tokens = []
        for match in self.pattern.findall(url):
            token = next((tok for tok in match if tok), None)
            if token:
                tokens.append(token)
        return tokens

    def build_vocab(self, urls):
        for url in urls:
            for tok in self.tokenize(url):
                if tok not in self.word2id:
                    idx = len(self.word2id)
                    self.word2id[tok] = idx
                    self.id2word[idx] = tok

    def encode(self, url):
        """Return CLS + token IDs."""
        tokens = self.tokenize(url)
        encoded = [self.word2id.get(tok, self.unk_id) for tok in tokens]
        return [self.cls_id] + encoded

    def decode(self, ids):
        return [self.id2word.get(i, "<unk>") for i in ids]

    @property
    def vocab_size(self):
        return len(self.word2id)