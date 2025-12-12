## Malicious Url Detection Deep Learning Model
Bi-LSTM and Bi-GRU model designed to detect malicious (malware, phishing, or defacement) or benign urls

## steps to train the model
### step 1. install dependencies
```bash
pip install -r requirements.txt
```

### step 2. set up Kaggle API (data acquisition.py needs this)
1. Go to [Kaggle Account Settings](https://www.kaggle.com/account)
2. Scroll to "API" section
3. Click "Create New Token" to download `kaggle.json`
4. Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows)

### step 3. acquire the data
```bash
python data_src/data_acquisition.py
```

### step 4. cmdline args to run train.py
train the model with the following cmd...
1. choose char or word for tokenizer_type
2. choose lstm or gru for rnn_type
```bash
python train.py \
  --data_path final_dataset.csv \
  --tokenizer_type char \
  --rnn_type lstm \
  --batch_size 32 \
  --max_len 256 \
  --epochs 50 \
  --lr 1e-3 \
  --dropout 0.3
```

### step 5. import load_model, load_tokenizer, and predict from inference.py 
```python
from inference import load_model, load_tokenizer, predict

# access the checkpoint from the lstm_birnn after training a bi-lstm model
tokenizer, _ = load_tokenizer('checkpoints/lstm_birnn/lstm_birnn_tokenizer.pt')  #lstm_birnn or gru_birnn is the folder produced when checkpoints.zip is unpacked
model = load_model('checkpoints/lstm_birnn/lstm_birnn-epoch=05-val_f1=0.9001.ckpt', 
                   model_type='birnn')

# access the checkpoint from the gru_birnn after training a bi-gru model
tokenizer, tok_type = load_tokenizer('checkpoints/gru_birnn/gru_birnn_tokenizer.pt')
model = load_model('checkpoints/gru_birnn/gru_birnn-epoch=02-val_f1=0.8968.ckpt',
                   model_type='birnn')
```

### step 6. save the model specifications after running train.py
for the google colab env
```python
import shutil
from google.colab import files

# Zip the checkpoints folder
shutil.make_archive('checkpoints', 'zip', 'checkpoints')

# Download the zipped file
files.download('checkpoints.zip')
```

for your local/remote machine
```python
import shutil, os

# Zip the checkpoints folder; creates checkpoints.zip in the current directory
zip_path = shutil.make_archive('checkpoints', 'zip', 'checkpoints')
print(f"Created zip at: {os.path.abspath(zip_path)}")
```


## steps to run the best model thus far (the pre-trained model)  🛑 only run this code if you've not trained the model; othwerise use step 5.
### step 1. install dependencies
```bash
pip install -r requirements.txt
```

### step 2. import load_model, load_tokenizer, and predict from inference.py
```python
from inference import load_model, load_tokenizer, predict

# The checkpoints were extracted directly into 'lstm_birnn' folder
tokenizer, _ = load_tokenizer('lstm_birnn/lstm_birnn_tokenizer.pt')
model = load_model('lstm_birnn/lstm_birnn-epoch=05-val_f1=0.9001.ckpt', 
                   model_type='birnn')

# The checkpoints were extracted directly into 'gru_birnn' folder
tokenizer, tok_type = load_tokenizer('gru_birnn/gru_birnn_tokenizer.pt')
model = load_model('gru_birnn/gru_birnn-epoch=10-val_f1=0.8942.ckpt',
                   model_type='birnn')               
```


## python for using specific prediction defintions found in inference.py
#### simple prediction
```python
prediction = predict("example.com", model, tokenizer)
print(prediction)  # Output: "Benign"

# with confidence scores
label, confidence, probs = predict("example.com", model, tokenizer, return_confidence=True)
print(f"Label: {label}")
print(f"Confidence: {confidence:.2%}")
print(f"Probabilities: {probs}")
```

#### url batch prediction
```python
urls = ["google.com", "example.com", "suspicious-site.com"]
predictions = predict(urls, model, tokenizer)
for url, pred in zip(urls, predictions):
    print(f"{url}: {pred}")
```

#### large url batch inference
```python
from inference import predict_batch

urls = [...]  # Large list of URLs
predictions = predict_batch(urls, model, tokenizer, batch_size=64)
view = zip(predictions, urls)
print(list(view))
```

## Running inference.py
#### cmdline args to run inference.py (if you've trained the model)
```bash
python inference.py \
  --checkpoint checkpoints/gru_birnn/gru_birnn-epoch=02-val_f1=0.8968.ckpt \
  --tokenizer checkpoints/gru_birnn/gru_birnn_tokenizer.pt \
  --model_type birnn \
  --urls "example.com" "google.com" \
  --confidence
```

#### cmdline args to run inference.py (if you're using the pre-trained model) 
#### 🛑 only works after running train.py; otherwise
#### you need to access gru_birnn or lstm_birnn from the checkpoints folder created by train.py
```bash
python inference.py \
  --checkpoint gru_birnn/gru_birnn-epoch=10-val_f1=0.8942.ckpt \
  --tokenizer gru_birnn/gru_birnn_tokenizer.pt \
  --model_type birnn \
  --urls "example.com" "google.com" \
  --confidence
```









### model scope & examples
- In-scope (what it was trained on): typical HTTP/HTTPS URLs that look like phishing, malware, defacement, or benign.  
  - `https://secure-paypal-login-verification.com/auth/update` (phishing-pattern hostname)  
  - `http://maliciousupdates.net/patches/system32_fix.exe` (malware-looking path/extension)  
  - `https://musicreviews.co/defacement/anonymous-ops/` (defacement-looking path)  
  - `https://example.com/profile` or `https://wikipedia.org/wiki/URL` (benign)
- Out-of-scope / likely poor performance: IP-host URLs, non-HTTP schemes, punycode tricks, heavily encoded Unicode, or URLs where key signals sit past 256 chars.  
  - IP host: `http://192.0.2.10/login`, `https://203.0.113.55/update.exe`  
  - Non-HTTP scheme: `ftp://downloads.example.com/app.exe`, `data:text/html,<script>alert(1)</script>`  
  - Punycode/IDN: `http://xn--pple-43d.com/login` (looks like “äpple.com”), `https://xn--googl-fsa.com/secure`  
  - Encoded Unicode/obfuscation: `https://login.microsoft.com/%e2%80%ae%e2%80%ae/account`, `http://example.com/%c3%a9%2Fverify`  
  - Overlength/obfuscated: `https://site.com/path/` followed by 300+ random chars where the malicious bits sit beyond position 256 (they’ll be truncated)
