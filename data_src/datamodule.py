"""
@ Description: retrieves all the url data, balances it among 4 classes, and stores it into a dataset file
@ Author: Kenisha Stills
@ Create Time: 11/09/25
"""

import lightning as L
from torch.utils.data import DataLoader, Subset
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from data_src.custom_tokenizers import (
    CharTokenizer,
    WordTokenizer,
)
from data_src.dataset import URLDataset

class URLDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        tokenizer_type: str = "char",
        max_len: int = 256,
        batch_size: int = 32,
        num_workers: int = 2,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
    ):
        super().__init__()
        self.data_path = data_path
        self.tokenizer_type = tokenizer_type
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Split proportions
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac

        # These will be created in setup()
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._split_indices = None

    # ------------------------------------------------------------------
    # prepare_data(): load raw CSV ONLY
    # ------------------------------------------------------------------
    def prepare_data(self):
        df = pd.read_csv(self.data_path)

        self.raw_urls = df["url"].astype(str).tolist()

        label_map = {
            "Benign": 0,
            "Phishing": 1,
            "Malware": 2,
            "Defacement": 3,
        }

        unique_labels = set(df["label"].unique())
        expected = set(label_map.keys())
        if not unique_labels.issubset(expected):
            raise ValueError(
                f"Unexpected labels found in dataset: {unique_labels - expected}. "
                f"Expected labels: {expected}"
            )

        self.raw_labels = [label_map[label] for label in df["label"]]

    def setup(self, stage=None):

        if self.train_dataset is not None and self.val_dataset is not None and self.test_dataset is not None:
            return

        dataset_size = len(self.raw_urls)

        if self._split_indices is None:
            # Use stratified splits to preserve class proportions
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=self.test_frac, random_state=42)
            train_val_idx, test_idx = next(splitter.split(self.raw_urls, self.raw_labels))
            train_val_idx = train_val_idx.tolist()
            test_idx = test_idx.tolist()

            val_ratio = self.val_frac / (self.train_frac + self.val_frac)
            val_split = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
            train_idx_rel, val_idx_rel = next(
                val_split.split([self.raw_urls[i] for i in train_val_idx],
                                [self.raw_labels[i] for i in train_val_idx])
            )
            train_idx = [train_val_idx[i] for i in train_idx_rel]
            val_idx = [train_val_idx[i] for i in val_idx_rel]
            
            self._split_indices = (train_idx, val_idx, test_idx)

        train_idx, val_idx, test_idx = self._split_indices

        if self.tokenizer is None:
            if self.tokenizer_type == "char":
                self.tokenizer = CharTokenizer()
            else:
                self.tokenizer = WordTokenizer()

            train_urls = [self.raw_urls[i] for i in train_idx]
            self.tokenizer.build_vocab(train_urls)
            
            # Log vocabulary statistics
            print(f"\n{'='*60}")
            print(f"Tokenizer: {self.tokenizer_type}")
            print(f"Vocabulary size: {self.tokenizer.vocab_size:,}")
            print(f"Training URLs: {len(train_urls):,}")
            print(f"Avg tokens per URL: {self._calculate_avg_tokens(train_urls):.1f}")
            print(f"{'='*60}\n")

        base_dataset = URLDataset(
            self.raw_urls,
            self.raw_labels,
            tokenizer=self.tokenizer,
            max_len=self.max_len,
        )

        self.train_dataset = Subset(base_dataset, train_idx)
        self.val_dataset = Subset(base_dataset, val_idx)
        self.test_dataset = Subset(base_dataset, test_idx)
        
        # Log split statistics
        self._log_split_statistics(train_idx, val_idx, test_idx)

    def _calculate_avg_tokens(self, urls):
        """Calculate average number of tokens per URL."""
        if not urls:
            return 0.0
        total_tokens = sum(len(self.tokenizer.encode(url)) for url in urls[:1000])  # Sample
        return total_tokens / min(1000, len(urls))
    
    def _log_split_statistics(self, train_idx, val_idx, test_idx):
        """Log dataset split statistics."""
        print(f"\n{'='*60}")
        print(f"Dataset Split Statistics")
        print(f"{'='*60}")
        print(f"Total samples: {len(self.raw_urls):,}")
        print(f"Training:   {len(train_idx):,} ({len(train_idx)/len(self.raw_urls)*100:.1f}%)")
        print(f"Validation: {len(val_idx):,} ({len(val_idx)/len(self.raw_urls)*100:.1f}%)")
        print(f"Test:       {len(test_idx):,} ({len(test_idx)/len(self.raw_urls)*100:.1f}%)")
        
        # Class distribution
        train_labels = [self.raw_labels[i] for i in train_idx]
        from collections import Counter
        label_counts = Counter(train_labels)
        
        print(f"\nTraining Class Distribution:")
        for label, count in sorted(label_counts.items()):
            print(f"  Class {label}: {count:,} ({count/len(train_labels)*100:.1f}%)")
        print(f"{'='*60}\n")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False, # speeds up trng
            pin_memory=True,  # Faster GPU transfer
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True,
        )