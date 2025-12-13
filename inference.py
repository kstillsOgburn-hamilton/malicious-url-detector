"""
@ Description: file to test the model's performance
@ Author: Kenisha Stills
@ Create Time: 11/17/25
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import argparse

from model_src.bilstm_gru import URLBiRNN
from data_src.custom_tokenizers import CharTokenizer, WordTokenizer

LABEL_MAP = {
    0: "Benign",
    1: "Phishing",
    2: "Malware",
    3: "Defacement",
}
# -------------------------------------------------------
#               LOAD TOKENIZER
# -------------------------------------------------------
def load_tokenizer(tokenizer_path="tokenizer.pt"):
    """
    Load the tokenizer saved by train.py.
    
    Args:
        tokenizer_path: Path to saved tokenizer file
        
    Returns:
        Tuple of (tokenizer, tokenizer_type)
        
    Raises:
        FileNotFoundError: If tokenizer file doesn't exist
        ValueError: If tokenizer file is corrupted or invalid
    """
    # Validate path exists
    if not Path(tokenizer_path).exists():
        raise FileNotFoundError(
            f"Tokenizer file not found: {tokenizer_path}\n"
            f"Please run training first to generate tokenizer."
        )
    
    try:
        data = torch.load(tokenizer_path, map_location="cpu")
    except Exception as e:
        raise ValueError(f"Failed to load tokenizer file: {e}")
    
    # Validate required keys
    required_keys = ["tokenizer_type", "stoi", "itos", "pad_id", "cls_id", "unk_id"]
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        raise ValueError(
            f"Tokenizer file is missing required keys: {missing_keys}"
        )

    tok_type = data["tokenizer_type"]

    # Create appropriate tokenizer
    if tok_type == "char":
        tok = CharTokenizer()
        tok.char2id = data["stoi"]
        tok.id2char = data["itos"]
    elif tok_type == "word":
        tok = WordTokenizer()
        tok.word2id = data["stoi"]
        tok.id2word = data["itos"]
    else:
        raise ValueError(f"Unknown tokenizer type: {tok_type}")

    tok.pad_id = data["pad_id"]
    tok.cls_id = data["cls_id"]
    tok.unk_id = data["unk_id"]

    return tok, tok_type


# -------------------------------------------------------
#               LOAD MODEL FROM CHECKPOINT
# -------------------------------------------------------
def load_model(checkpoint_path, model_type, device=None):
    """
    Loads Transformer, BiRNN, or Hybrid model from a Lightning checkpoint.
    Uses hyperparameters stored in the checkpoint (via save_hyperparameters).
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_type: "transformer", "birnn", or "hybrid"
        device: Device to load model on (default: auto-detect)
        
    Returns:
        Loaded model in eval mode
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}\n"
            f"Please train a model first."
        )
    
    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
        hyper_params = ckpt["hyper_parameters"]
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint file: {e}")

   
    if model_type == "birnn":
        model = URLBiRNN(**hyper_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'transformer', 'birnn', or 'hybrid'")

    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


# -------------------------------------------------------
#               PREPROCESS SINGLE URL
# -------------------------------------------------------
def preprocess(url, tokenizer, max_len=256):
    import re

    url = url.lower().strip()
    url = re.sub(r'^(https?://)', '', url)
    url = re.sub(r'^(www\.)', '', url)
    url = re.sub(r'/$', '', url)

    ids = tokenizer.encode(url)
    ids = ids[:max_len]
    pad_len = max_len - len(ids)

    ids = F.pad(
        torch.tensor(ids, dtype=torch.long),
        (0, pad_len),
        value=tokenizer.pad_id,
    )

    return ids.unsqueeze(0)


# -------------------------------------------------------
#               PREDICT
# -------------------------------------------------------
def predict(urls, model, tokenizer, device=None, return_confidence=False):
    """
    Predict labels for single URL or batch of URLs.
    
    Args:
        urls: Single URL string or list of URL strings
        model: Loaded model
        tokenizer: Loaded tokenizer
        device: Device to run inference on (default: auto-detect)
        return_confidence: If True, return (label, confidence, all_probs)
        
    Returns:
        If return_confidence=False:
            - Single URL: predicted label string
            - Batch: list of predicted label strings
        If return_confidence=True:
            - Single URL: tuple of (label, confidence, class_probabilities)
            - Batch: list of tuples
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    
    # Handle single URL vs batch
    is_single = isinstance(urls, str)
    if is_single:
        urls = [urls]
    
    if not urls:
        return [] if not is_single else None
    
    max_len = model.hparams.max_len
    batch = torch.stack([preprocess(u, tokenizer, max_len)[0] for u in urls]).to(device)

    pad_id = tokenizer.pad_id
    lengths = []
    for row in batch:
        non_pad = (row != pad_id).nonzero(as_tuple=True)[0]
        lengths.append(int(non_pad[-1]) + 1 if len(non_pad) else 1)
    lengths = torch.tensor(lengths, dtype=torch.long, device=device)

    with torch.no_grad():
        if hasattr(model.hparams, "hidden_dim"):
            logits = model(batch, lengths.cpu())
        else:
            logits = model(batch)
        
        # Get probabilities
        probs = torch.softmax(logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)
        
        pred_labels = [LABEL_MAP.get(p.item(), "Unknown") for p in pred]
        confidences = confidence.tolist()
        
        if return_confidence:
            # Return label, confidence, and all class probabilities
            all_probs = [
                {LABEL_MAP[i]: probs[j, i].item() for i in range(len(LABEL_MAP))}
                for j in range(len(pred_labels))
            ]
            if is_single:
                return pred_labels[0], confidences[0], all_probs[0]
            return list(zip(pred_labels, confidences, all_probs))
        
        return pred_labels[0] if is_single else pred_labels


# -------------------------------------------------------
#               BATCH PREDICT (Efficient for large batches)
# -------------------------------------------------------
def predict_batch(urls, model, tokenizer, batch_size=32, device=None):
    """
    Predict labels for multiple URLs efficiently in batches.
    
    Args:
        urls: List of URL strings
        model: Loaded model
        tokenizer: Loaded tokenizer
        batch_size: Batch size for inference
        device: Device to run inference on (default: auto-detect)
        
    Returns:
        List of predicted labels
    """
    if not urls:
        return []
    
    if not isinstance(urls, list):
        urls = [urls]
    
    all_predictions = []
    
    # Process in batches
    for i in range(0, len(urls), batch_size):
        batch_urls = urls[i:i + batch_size]
        batch_preds = predict(batch_urls, model, tokenizer, device=device, return_confidence=False)
        all_predictions.extend(batch_preds if isinstance(batch_preds, list) else [batch_preds])
    
    return all_predictions
# -------------------------------------------------------
#               MAIN EXECUTION BLOCK
# -------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='URL Maliciousness Prediction')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint file.')
    parser.add_argument('--tokenizer', type=str, required=True, help='Path to tokenizer file.')
    parser.add_argument('--model_type', type=str, required=True, choices=['birnn', 'transformer', 'hybrid'], help='Type of model used (birnn, transformer, hybrid).')
    parser.add_argument('--urls', nargs='+', required=True, help='One or more URLs to predict.')
    parser.add_argument('--confidence', action='store_true', help='Return confidence scores and all probabilities.')
    parser.add_argument('--device', type=str, default=None, help='Device to run inference on (e.g., cuda:0, cpu).')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for prediction (if multiple URLs).')

    args = parser.parse_args()

    try:
        tokenizer, _ = load_tokenizer(args.tokenizer)
        model = load_model(args.checkpoint, args.model_type, device=args.device)

        predictions = predict(args.urls, model, tokenizer, device=args.device, return_confidence=args.confidence)

        if args.confidence:
            for url, (label, conf, probs) in zip(args.urls, predictions):
                print(f"URL: {url}")
                print(f"  Prediction: {label}")
                print(f"  Confidence: {conf:.2%}")
                print("  All probabilities:")
                for class_name, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                    print(f"    {class_name}: {prob:.2%}")
                print("-" * 30)
        else:
            for url, label in zip(args.urls, predictions):
                print(f"URL: {url}, Prediction: {label}")

    except Exception as e:
        print(f"Error during inference: {e}")
        import sys
        sys.exit(1)
