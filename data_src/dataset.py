"""
@ Description: retrieves all the url data, balances it among 4 classes, and stores it into a dataset file
@ Author: Kenisha Stills
@ Create Time: 11/09/25
"""

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class URLDataset(Dataset):
    """
    Dataset for URL classification with automatic tokenization and padding.
    
    Args:
        urls: List of URL strings
        labels: List of integer labels (0-3)
        tokenizer: CharTokenizer or WordTokenizer instance
        max_len: Maximum sequence length for padding/truncation
        
    Raises:
        ValueError: If inputs are invalid
    """
    
    def __init__(self, urls, labels, tokenizer, max_len):
        # Validate inputs
        if not urls or not labels:
            raise ValueError("URLs and labels cannot be empty")
        
        if len(urls) != len(labels):
            raise ValueError(
                f"URLs and labels must have same length: "
                f"got {len(urls)} URLs and {len(labels)} labels"
            )
        
        if max_len <= 0:
            raise ValueError(f"max_len must be positive, got {max_len}")
        
        if not hasattr(tokenizer, 'encode') or not hasattr(tokenizer, 'pad_id'):
            raise ValueError("tokenizer must have 'encode' and 'pad_id' attributes")
        
        self.urls = urls
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = self.tokenizer.pad_id
        
    def __getitem__(self, idx):
        """
        Get single URL, label, and length tuple.
        
        Args:
            idx: Index of item to retrieve
            
        Returns:
            Tuple of (tokens, label, original_length)
            - tokens: LongTensor of shape (max_len,)
            - label: LongTensor scalar
            - original_length: LongTensor scalar
            
        Raises:
            Exception: Re-raises with context if tokenization fails
        """
        try:
            url = self.urls[idx]
            
            # Handle potential non-string URLs
            if not isinstance(url, str):
                url = str(url)
            
            tokens = self.tokenizer.encode(url)
            
            # 1. Truncation
            original_len = len(tokens)
            if original_len > self.max_len:
                tokens = tokens[:self.max_len]
                original_len = self.max_len
            
            # Edge case: empty URL -> at least CLS token
            if original_len == 0:
                tokens = [self.tokenizer.cls_id]
                original_len = 1
            
            # 2. Padding
            pad_len = self.max_len - len(tokens)
            
            if pad_len > 0:
                tokens = F.pad(
                    torch.tensor(tokens, dtype=torch.long),
                    (0, pad_len),
                    value=self.pad_id
                )
            else:
                tokens = torch.tensor(tokens, dtype=torch.long)
            
            return (
                tokens,
                torch.tensor(self.labels[idx], dtype=torch.long),
                torch.tensor(original_len, dtype=torch.long)
            )
            
        except Exception as e:
            # Provide context for debugging
            raise RuntimeError(
                f"Error processing URL at index {idx}: '{url}' - {str(e)}"
            ) from e
        
    def __len__(self):
        return len(self.urls)