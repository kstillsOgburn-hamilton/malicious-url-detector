"""
@ Description: custom tokenizers for the url's
@ Author: Kenisha Stills
@ Create Time: 11/10/25
"""

import torch
import torch.nn as nn
import lightning as L
from torchmetrics import Accuracy, F1Score
from torchmetrics.classification import ConfusionMatrix
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import wandb

class URLBiRNN(L.LightningModule):
    """
    Bidirectional RNN for classifying URLs into one of `num_classes` categories.

    Architecture:
        1. Embedding Layer: Maps token IDs to dense vectors.
        2. Bi-RNN (LSTM/GRU): Processes sequence context in both directions.
           - Uses `pack_padded_sequence` for efficient masking.
        3. Pooling: Extracts the first token's hidden state (index 0).
        4. Classifier: Dropout -> Linear Layer -> Logits.

    Args:
        vocab_size (int): Size of the token vocabulary.
        embed_dim (int): Dimension of the dense embedding vectors. Defaults to 128.
        hidden_dim (int): Dimension of the RNN hidden state (per direction). 
                          The concatenated output will be hidden_dim * 2. Defaults to 256.
        rnn_type (str): The architecture to use ('lstm' or 'gru'). Defaults to 'lstm'.
        num_layers (int): Number of stacked RNN layers. Defaults to 2.
        num_classes (int): Number of target classes. Defaults to 4.
        dropout (float): Dropout probability applied to the pooled vector before classification. Defaults to 0.3.
        lr (float): Initial learning rate for the AdamW optimizer. Defaults to 1e-3.
        pad_token_id (int): The specific token ID used for padding (ignored by embedding). Defaults to 0.
        max_len (int): Maximum sequence length; used to ensure consistent tensor shapes after unpacking. Defaults to 256.
        total_epochs (int): Total training epochs; used to calculate T_max for the Cosine Annealing scheduler. Defaults to 10.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        rnn_type: str = "lstm",   # "lstm" or "gru"
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.3,
        lr: float = 1e-3,
        pad_token_id: int = 0,
        max_len: int = 256,
        total_epochs: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.pad_id = pad_token_id

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=self.pad_id,
        )

        if rnn_type.lower() == "gru":
            self.rnn = nn.GRU(
                embed_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
            )
        else:
            self.rnn = nn.LSTM(
                embed_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
            )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()
        # Separate metric instances per split so states do not bleed across phases
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_cm = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.label_names = ["Benign", "Phishing", "Malware", "Defacement"]

    def forward(self, x_padded, x_lengths):
        emb = self.embedding(x_padded)              # (batch, seq, embed_dim)

        emb_packed = pack_padded_sequence(
            emb, 
            x_lengths.cpu(), # lengths must be on CPU
            batch_first=True, 
            enforce_sorted=False
        )
        rnn_out_packed, _ = self.rnn(emb_packed)

        rnn_out, _ = pad_packed_sequence(
            rnn_out_packed, 
            batch_first=True,
            total_length=self.hparams.max_len # Pad back to max_len
        )          # (batch, seq, hidden*2)

        # CLS pooling (first token)
        cls_vec = rnn_out[:, 0, :]           # (batch, hidden*2)
        cls_vec = self.dropout(cls_vec)

        logits = self.classifier(cls_vec)    # (batch, num_classes)
        return logits

    def training_step(self, batch, batch_idx):
        x, y, x_lengths = batch
        logits = self(x, x_lengths)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.train_acc(preds, y)
        self.train_f1(preds, y)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, x_lengths = batch
        logits = self(x, x_lengths)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_f1", self.val_f1, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y, x_lengths = batch
        logits = self(x, x_lengths)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        self.test_cm.update(preds, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True)

    def on_test_epoch_end(self):
        cm = self.test_cm.compute()
        if isinstance(cm, torch.Tensor):
            cm = cm.detach().cpu().numpy()

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.label_names)
        disp.plot(cmap="Blues", colorbar=False)
        fig = disp.figure_
        for logger in self.trainer.loggers or []:
            if isinstance(logger, WandbLogger):
                logger.experiment.log({"confusion_matrix": wandb.Image(fig)})
            elif isinstance(logger, TensorBoardLogger):
                logger.experiment.add_figure("confusion_matrix", fig, global_step=self.current_epoch)
        plt.close(fig)
        self.test_cm.reset()

    def configure_optimizers(self):
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=0.01,
            )
            # Access the max_epochs from the trainer
            # Use a fallback (e.g., 10) in case self.trainer is not available
            t_max = self.trainer.max_epochs if self.trainer else 10
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.total_epochs, # <-- Use the hparam
                eta_min=self.hparams.lr / 50,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
