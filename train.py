"""
@ Description: file to train the model
@ Author: Kenisha Stills
@ Create Time: 11/15/25
"""

import argparse
from datetime import datetime
from pathlib import Path

import lightning as L
import torch
import wandb

from data_src.datamodule import URLDataModule
from model_src.bilstm_gru import URLBiRNN

from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

# ---------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train URL Classification Models")

    # Data
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--tokenizer_type", type=str, default="char",
                        choices=["char", "word"])

    # RNN hyperparameters
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_rnn_layers", type=int, default=2)
    parser.add_argument("--rnn_type", type=str, default="lstm",
                        choices=["lstm", "gru"])
    parser.add_argument("--dropout", type=float, default=0.3)

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)

    return parser.parse_args()

# ---------------------------------------------------------
# Main training function
# ---------------------------------------------------------
def main():
    args = parse_args()
    # -----------------------------
    # DataModule
    # -----------------------------
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    dm = URLDataModule(
        data_path=args.data_path,
        tokenizer_type=args.tokenizer_type,
        batch_size=args.batch_size,
        max_len=args.max_len,
    )

    # We call these explicitly so we can access tokenizer to get vocab_size/pad_id.
    dm.prepare_data()
    dm.setup()

    # Determine vocab size from tokenizer
    vocab_size = dm.tokenizer.vocab_size
    pad_id = dm.tokenizer.pad_id

    # -----------------------------
    # BiRNN Model
    # -----------------------------
    model = URLBiRNN(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_rnn_layers,
            num_classes=4,
            dropout=args.dropout,
            lr=args.lr,
            rnn_type=args.rnn_type,
            pad_token_id=pad_id,
            max_len=args.max_len,
            total_epochs=args.epochs
        )
    model_name = f"{args.rnn_type}_birnn"

    tb_logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name=model_name,
        version=None  # timestamped folders
    )

    wandb_logger = WandbLogger(
        project="malicious-url-classifier",  # adjust to your W&B project
        config=vars(args),  # log hyperparameters
        log_model=True,     # uploads best checkpoints
        name=f"{model_name}_{timestamp}" # deterministic name on W&B

    )
    wandb_logger.watch(model, log="all", log_freq=50)

    # -----------------------------
    # Callbacks
    # -----------------------------
    early_stop = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=5,
        min_delta=0.001,
    )

    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints") / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        monitor="val_f1",
        mode="max",
        filename=f"{model_name}" + "-{epoch:02d}-{val_f1:.4f}",
        save_top_k=3,
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # -----------------------------
    # Save tokenizer for inference (co-locate with checkpoint)
    # -----------------------------
    tokenizer_save_path = checkpoint_dir / f"{model_name}_tokenizer.pt"
    torch.save(
        {
            "stoi": dm.tokenizer.char2id if args.tokenizer_type == "char" else dm.tokenizer.word2id,
            "itos": dm.tokenizer.id2char if args.tokenizer_type == "char" else dm.tokenizer.id2word,
            "pad_id": dm.tokenizer.pad_id,
            "cls_id": dm.tokenizer.cls_id,
            "unk_id": dm.tokenizer.unk_id,
            "tokenizer_type": args.tokenizer_type,
            "vocab_size": vocab_size,
        },
        str(tokenizer_save_path),
    )
    print(f"Tokenizer saved to: {tokenizer_save_path}")

    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[early_stop, checkpoint_cb, lr_monitor],
        logger=[wandb_logger, tb_logger]
    )
    # -----------------------------
    # Train
    # -----------------------------
    trainer.fit(model, dm)
    # -----------------------------
    # Test (logs to WandB/TensorBoard)
    # -----------------------------
    trainer.test(ckpt_path="best", datamodule=dm)

if __name__ == "__main__":
    main()
