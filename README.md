## Malicious URL Detection Deep Learning Model
Bi-LSTM and Bi-GRU model designed to detect malicious (malware, phishing, or defacement) or benign urls

## download the malicious-url-detector repo and setup environment
### step 1. clone the repo
```bash
git clone https://github.com/kstillsOgburn-hamilton/malicious-url-detector.git
```
### step 2. create a virtual environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n url_classifier python=3.10 # if 3.10 doesn't work try 3.12
conda activate url_classifier
```

### step 3. install dependencies
```bash
pip install -r requirements.txt
```

## steps to train the model
### step 4. set up Kaggle API (data acquisition.py needs this)
1. Create a Kaggle account if you don't already have one.
2. Go to [Kaggle Account Settings](https://www.kaggle.com/account)
3. Scroll to "API" section
4. Click "Create New Token" to download `kaggle.json`
5. Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows)

### step 5. acquire and clean the data
```bash
python data_src/data_acquisition.py
```

### step 6. cmdline args to run train.py
train the model with the following cmd...
1. get the absolute path to the `final_dataset.csv`
2. choose char or word for tokenizer_type
3. choose lstm or gru for rnn_type
4. choose the other hyperparameter args (i.e. batch size, etc.)
```bash

BASIC OPTION
python train.py \
  --data_path final_dataset.csv \ # the file path for the dataset
  --tokenizer_type char \
  --hidden_dim 64 \
  --rnn_type gru \
  --batch_size 32 \
  --max_len 256 \
  --epochs 50 \
  --lr 1e-3 \
  --dropout 0.2


COMPREHENSIVE OPTION
python train.py \
  --data_path final_dataset.csv \ 
  --tokenizer_type word \ 
  --rnn_type gru \ 
  --embed_dim 128 \ 
  --hidden_dim 256 \ 
  --num_rnn_layers 2 \ 
  --dropout 0.3 \ 
  --batch_size 64 \ 
  --max_len 256 \ 
  --epochs 50 \ 
  --lr 1e-3.0
```
After training, you'll find:
- **Checkpoints**: `checkpoints/{model_name}/`
  - Best model: `{model_name}-epoch={epoch}-val_f1={f1}.ckpt`
  - Last model: `last.ckpt`
- **Tokenizer**: `checkpoints/{model_name}/{model_name}_tokenizer.pt`

### step 7. import load_model, load_tokenizer, and predict from inference.py 
```python
from inference import load_model, load_tokenizer, predict

# access the checkpoint from the checkpoints/lstm_birnn folder after training a bi-lstm model
tokenizer, _ = load_tokenizer('checkpoints/lstm_birnn/lstm_birnn_tokenizer.pt')  # lstm_birnn or gru_birnn is the folder produced when checkpoints.zip is unzipped
model = load_model('checkpoints/lstm_birnn/lstm_birnn-epoch=05-val_f1=0.9001.ckpt', 
                   model_type='birnn')

# access the checkpoint from the checkpoints/gru_birnn folder after training a bi-gru model
tokenizer, tok_type = load_tokenizer('checkpoints/gru_birnn/gru_birnn_tokenizer.pt')
model = load_model('checkpoints/gru_birnn/gru_birnn-epoch=02-val_f1=0.8968.ckpt',
                   model_type='birnn')
```

### step 7. locate the model and tokenizer after running train.py
for your local/remote machine
#### After training, you'll find the model was already saved here...
`checkpoints/{model_name}/{model_name}-epoch={epoch}-val_f1={f1}.ckpt`
#### and the tokenizer was saved here...
`checkpoints/{model_name}/{model_name}_tokenizer.pt`


## steps to run the trained model 
## 🛑 only run this code if you want to use the TRAINED model; othwerise go back to step 6 to load the latest model

### step 1. import load_model, load_tokenizer, and predict from inference.py
```python
from inference import load_model, load_tokenizer, predict

# Extract the checkpoints.zip file to access the 'lstm_birnn' folder directly
tokenizer, _ = load_tokenizer('lstm_birnn/lstm_birnn_tokenizer.pt')
model = load_model('lstm_birnn/lstm_birnn-epoch=05-val_f1=0.9001.ckpt', 
                   model_type='birnn')

# Extract the checkpoints.zip file to access the 'gru_birnn' folder directly
tokenizer, tok_type = load_tokenizer('gru_birnn/gru_birnn_tokenizer.pt')
model = load_model('gru_birnn/gru_birnn-epoch=10-val_f1=0.8942.ckpt',
                   model_type='birnn')  
```

## python code to use specific prediction defintions found in inference.py

#### single url prediction
```python
url = "hamilton.edu" # sample url
prediction = predict(url, model, tokenizer)
print(prediction)  # Output: "Benign"

# with confidence scores
label, confidence, probs = predict(url, model, tokenizer, return_confidence=True)
print(f"URL: {url}")
print(f"Prediction: {label}")
print(f"Confidence: {confidence:.2%}")
print(f"All probabilities:")
for class_name, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
    print(f"  {class_name}: {prob:.2%}")
```

#### small url batch prediction
```python
urls = ["hamilton.edu", "www.hamilton.edu", "https://www.hamilton.edu/",]
predictions = predict(urls, model, tokenizer)
for url, pred in zip(urls, predictions):
    print(f"{url}: {pred}")
```

#### large url batch inference (1)
```python
from inference import predict_batch

urls = [...]  # Large list of URLs
predictions = predict_batch(urls, model, tokenizer, batch_size=64)
view = zip(predictions, urls)
for pred, url in view:
    print(f"('{url}', '{pred}')")
```

#### large url batch inference (2)
```python
for url in urls:
    label, confidence, probs = predict(url, model, tokenizer, return_confidence=True)
    print(f"URL: {url}")
    print(f"  Prediction: {label}")
    print(f"  Confidence: {confidence:.2%}")
    print("  All probabilities:")
    for class_name, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {prob:.2%}")
    print("------------------------------")
```

## Running inference.py
#### cmdline args to run inference.py AFTER training the model
```bash
python inference.py \
  --checkpoint checkpoints/gru_birnn/gru_birnn-epoch=02-val_f1=0.8942.ckpt \
  --tokenizer checkpoints/gru_birnn/gru_birnn_tokenizer.pt \
  --model_type birnn \
  --urls "youtube.com" "google.com" \
  --confidence
```

#### cmdline args to run inference.py on the TRAINED model. 🛑 unzipping the checkpoints.zip will produce the gru_birnn folder or the lstm_birnn folder, depending on the RNN model used during training (right now gru_birnn has already been downloaded from checkpoints.zip and exists in .../malicious-url-detector directory)
```bash
python inference.py \
  --checkpoint gru_birnn/gru_birnn-epoch=10-val_f1=0.8942.ckpt \ # currenlty using the best model
  --tokenizer gru_birnn/gru_birnn_tokenizer.pt \
  --model_type birnn \
  --urls "www.hamilton.edu" "google.com" \
  --confidence
```


### model scope & examples
In-scope (what it was trained on): URL strings (normalized by lowercasing and stripping http(s)://, leading www., and trailing /) that resemble benign, phishing, malware, or defacement patterns in the hostname/path.
Examples (shown as full URLs, but normalized before tokenization):

https://secure-paypal-login-verification.com/auth/update (phishing-like hostname tokens)

http://maliciousupdates.net/patches/system32_fix.exe (malware-like path/extension)

https://musicreviews.co/defacement/anonymous-ops/ (defacement-like path tokens)

https://wikipedia.org/wiki/URL (benign-like)

Out-of-scope / likely poor performance: IP-host URLs (removed during dataset cleaning), non-HTTP schemes (rare/absent in training and not stripped by normalization), IDN/punycode tricks, heavy encoding/Unicode obfuscation, and cases where key signals occur beyond 256 tokens and are truncated.

IP host: http://192.0.2.10/login, https://203.0.113.55/update.exe

Non-HTTP: ftp://downloads.example.com/app.exe, data:text/html,<script>…</script>

Punycode/IDN: http://xn--pple-43d.com/login

Encoded/obfuscated: https://login.microsoft.com/%e2%80%ae…

Overlength: URLs where malicious indicators appear only after the first 256 tokens (truncated)