"""
@ Description: retrieves all the url data, balances it among 4 classes, and stores it into a dataset file
@ Author: Kenisha Stills
@ Create Time: 11/07/25
"""

import pandas as pd
import requests
import io
import tranco  # pip install tranco
import kagglehub  # pip install kagglehub
import os
import zipfile
import re
from urllib.parse import urlparse

# --- 1. Configuration & Labels ---
KAGGLE_DATASET_NAME = "sid321axn/malicious-urls-dataset"
KAGGLE_CSV_NAME = "malicious_phish.csv"
URLHAUS_URL = "https://urlhaus.abuse.ch/downloads/csv_online/"
OPENPHISH_URL = "https://raw.githubusercontent.com/openphish/public_feed/refs/heads/main/feed.txt"
FINAL_OUTPUT_FILE = "final_dataset.csv"

LABEL_BENIGN = "Benign"
LABEL_DEFACE = "Defacement"
LABEL_PHISHING = "Phishing"
LABEL_MALWARE = "Malware"

# --- 2. Data Fetching Functions ---
def load_kaggle_data():
    print(f"Loading Kaggle dataset: {KAGGLE_DATASET_NAME}...")
    try:
        dataset_path = kagglehub.dataset_download(KAGGLE_DATASET_NAME)
        if dataset_path.endswith(".zip"):
            extract_path = os.path.join(os.path.dirname(dataset_path), "kaggle_data_unzipped")
            os.makedirs(extract_path, exist_ok=True)
            with zipfile.ZipFile(dataset_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)
            csv_file_path = os.path.join(extract_path, KAGGLE_CSV_NAME)
        else:
            csv_file_path = os.path.join(dataset_path, KAGGLE_CSV_NAME)

        if not os.path.exists(csv_file_path):
            return pd.DataFrame({"url": [], "label": []})

        df = pd.read_csv(csv_file_path)
        df = df[["url", "type"]]
        df.columns = ["url", "label"]
        df["label"] = df["label"].replace({
            "benign": LABEL_BENIGN, "defacement": LABEL_DEFACE,
            "phishing": LABEL_PHISHING, "malware": LABEL_MALWARE,
        })
        return df
    except Exception as e:
        print(f"Error loading Kaggle: {e}", exc_info=True)
        return pd.DataFrame({"url": [], "label": []})

def fetch_urlhaus_data():
    print("Fetching URLhaus data (Malware)...")
    try:
        r = requests.get(URLHAUS_URL, timeout=60)
        csv_content = io.StringIO(r.content.decode("utf-8", errors="ignore"))
        df = pd.read_csv(csv_content, skiprows=9, header=None, names=["id","dateadded","url","url_status","last_online","threat","tags","urlhaus_link","reporter"], quotechar='"')
        df = df[df["url_status"] == "online"].copy()
        df["label"] = LABEL_MALWARE
        return df[["url", "label"]]
    except Exception as e:
        print(f"Error URLhaus: {e}", exc_info=True)
        return pd.DataFrame({"url": [], "label": []})

def fetch_openphish_data():
    print("Fetching OpenPhish data (Phishing)...")
    try:
        r = requests.get(OPENPHISH_URL, timeout=30)
        urls = [u.strip() for u in r.text.split("\n") if u.strip()]
        df = pd.DataFrame(urls, columns=["url"])
        df["label"] = LABEL_PHISHING
        return df
    except Exception as e:
        print(f"Error OpenPhish: {e}", exc_info=True)
        return pd.DataFrame({"url": [], "label": []})

def fetch_tranco_data(limit=None):
    print("Fetching Tranco data (Benign)...")
    try:
        t = tranco.Tranco(cache=True, cache_dir=".tranco_cache")
        domains = list(t.list().list)
        if limit: domains = domains[:limit]
        df = pd.DataFrame(domains, columns=["domain"])
        df["url"] = "http://" + df["domain"]
        df["label"] = LABEL_BENIGN
        return df[["url", "label"]]
    except Exception as e:
        print(f"Error Tranco: {e}", exc_info=True)
        return pd.DataFrame({"url": [], "label": []})

# --- 3. Cleaning Helpers ---
def normalize_url(url):
    """
    Normalize a single URL for consistent processing.
    
    Args:
        url: Raw URL string
        
    Returns:
        Normalized URL string (lowercase, no protocol/www, no trailing slash)
    """
    if not isinstance(url, str):
        return ""
    
    url = url.lower().strip()
    # Remove protocol
    url = re.sub(r'^(https?://)', '', url)
    # Remove www
    url = re.sub(r'^(www\.)', '', url)
    # Remove trailing slash
    url = re.sub(r'/$', '', url)
    
    return url

def host_is_ip(url_str):
    """Check if URL hostname is an IPv4 address."""
    try:
        # Handle invalid URLs gracefully
        if not isinstance(url_str, str) or not url_str.strip():
            return False
        
        # Add protocol if missing for proper parsing
        if "://" not in url_str:
            url_str = f"http://{url_str}"
        
        # Parse URL - handle potential ValueError from invalid URLs
        try:
            parsed = urlparse(url_str)
        except (ValueError, Exception):
            # If URL is invalid, assume it's not an IP address
            return False
        
        # Extract hostname (remove port if present)
        host = parsed.netloc.split(":")[0]
        
        # Check if hostname matches IPv4 pattern
        ip_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
        return bool(re.fullmatch(ip_pattern, host))
    except Exception:
        # If anything goes wrong, assume it's not an IP address
        return False

def remove_ip_addresses(df):
    """Removes rows where the URL hostname is an IPv4 address."""
    initial = len(df)
    df_clean = df[~df['url'].astype(str).apply(host_is_ip)]
    print(f"Removed {initial - len(df_clean)} URLs with IP address hostnames.")
    return df_clean

def global_deduplication(df):
    """Normalizes URLs and removes duplicates."""
    initial = len(df)
    
    # Use the new normalize_url function
    df["url_key"] = df["url"].astype(str).apply(normalize_url)
    
    df_clean = df.drop_duplicates(subset=["url_key"], keep="first")
    df_clean = df_clean.drop(columns=["url_key"])
    
    print(f"Global Deduplication: Removed {initial - len(df_clean)} duplicates.")
    return df_clean

# --- 4. Data Validation ---
def validate_dataframe(df, source_name, min_rows=100):
    """
    Validate fetched dataframe has required structure and minimum data.
    
    Args:
        df: DataFrame to validate
        source_name: Name of data source for error messages
        min_rows: Minimum required rows
        
    Returns:
        bool: True if valid
        
    Raises:
        ValueError: If validation fails
    """
    if df is None or len(df) == 0:
        raise ValueError(f"{source_name}: No data fetched")
    
    if not all(col in df.columns for col in ['url', 'label']):
        raise ValueError(f"{source_name}: Missing required columns (url, label)")
    
    if len(df) < min_rows:
        print(f"{source_name}: Only {len(df)} rows (expected >={min_rows})")
    
    # Check for null URLs
    null_count = df['url'].isnull().sum()
    if null_count > 0:
        print(f"{source_name}: {null_count} null URLs will be removed")
    
    return True

# --- 5. Main Execution ---
def main(): 
    print("--- Phase 1: Data Acquisition Started ---")
    
    try:
        # 1. Fetch Raw Data
        df_k = load_kaggle_data()
        validate_dataframe(df_k, "Kaggle", min_rows=1000)
        
        df_u = fetch_urlhaus_data()
        validate_dataframe(df_u, "URLhaus", min_rows=100)
        
        df_o = fetch_openphish_data()
        validate_dataframe(df_o, "OpenPhish", min_rows=100)
        
        df_t = fetch_tranco_data()
        validate_dataframe(df_t, "Tranco", min_rows=1000)
        
    except ValueError as e:
        print(f"Data validation failed: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error during data acquisition: {e}", exc_info=True)
        raise

    # --- 2. Pool Individual Classes (Restored Logic) ---
    
    # A. BENIGN (Kaggle + Tranco)
    print("\nAssembling Benign Pool...")
    df_k_benign = df_k[df_k['label'] == LABEL_BENIGN].copy()

    tranco_sample_size = min(len(df_k_benign), len(df_t))
    df_t_sample = df_t.sample(n=tranco_sample_size, random_state=42)

    df_benign_pool = pd.concat([df_k_benign, df_t_sample], ignore_index=True)
    print(f"  Benign Pool: {len(df_benign_pool)} URLs")

    # Augment benign with www-prefixed variants
    benign_www = df_benign_pool.copy()
    benign_www['url'] = 'www.' + benign_www['url'].astype(str).str.replace(r'^www\.', '', regex=True)
    df_benign_pool = pd.concat([df_benign_pool, benign_www], ignore_index=True)


    # B. MALWARE (Kaggle + URLhaus)
    print("Assembling Malware Pool...")
    df_k_malware = df_k[df_k['label'] == LABEL_MALWARE]
    df_malware_pool = pd.concat([df_k_malware, df_u], ignore_index=True)
    print(f"  Malware Pool: {len(df_malware_pool)} URLs")

    # C. PHISHING (Kaggle + OpenPhish)
    print("Assembling Phishing Pool...")
    df_k_phish = df_k[df_k['label'] == LABEL_PHISHING]
    df_phish_pool = pd.concat([df_k_phish, df_o], ignore_index=True)
    print(f"  Phishing Pool: {len(df_phish_pool)} URLs")

    # D. DEFACEMENT (Kaggle only)
    print("Assembling Defacement Pool...")
    df_deface_pool = df_k[df_k['label'] == LABEL_DEFACE].copy()
    print(f"  Defacement Pool: {len(df_deface_pool)} URLs")

    # --- 3. Combine & Clean (New Logic) ---
    
    print("\nCombining all pools...")
    # Combine everything BEFORE deduping
    full_df = pd.concat([df_benign_pool, df_malware_pool, df_phish_pool, df_deface_pool], ignore_index=True)
    full_df.dropna(subset=["url"], inplace=True)
    print(f"Total Raw Count: {len(full_df)}")

    # A. Remove IP Addresses (Requested)
    print("\nCleaning IP Addresses...")
    full_df = remove_ip_addresses(full_df)

    # B. Global Deduplication (Requested)
    print("Performing Global Deduplication...")
    full_df = global_deduplication(full_df)

    # --- 4. Balance Classes ---
    print("\n--- Balancing Classes ---")
    counts = full_df['label'].value_counts()
    print(f"\n{counts}")
    
    # Validate all expected classes are present
    expected_classes = {LABEL_BENIGN, LABEL_DEFACE, LABEL_PHISHING, LABEL_MALWARE}
    missing = expected_classes - set(counts.index)
    if missing:
        raise RuntimeError(
            f"Missing classes after acquisition: {missing}. "
            f"Pipeline cannot proceed without all classes. Check data sources."
        )
    
    min_count = counts.min()
    print(f"\nBalancing to minority class count: {min_count}")
    
    # --- NEW: Loop approach with validation ---
    dfs = []
    for label, df_class in full_df.groupby('label'):
        if len(df_class) < min_count:
            raise RuntimeError(
                f"Not enough samples for {label}: {len(df_class)} < {min_count}. "
                f"Cannot balance dataset. Check data sources or reduce min_count."
            )
        dfs.append(df_class.sample(n=min_count, random_state=42))
        
    # Combine and shuffle
    balanced_df = pd.concat(dfs, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Balanced dataset size: {balanced_df.size}")
    
    # --- 5. Save ---
    # Add this line to clean the protocols from the final 'url' column
    print("Removing http/https protocols and leading www from final URL list...")
    balanced_df['url'] = (
    balanced_df['url']
    .astype(str)
    .str.replace(r"^(http://|https://)", "", regex=True)
    .str.replace(r"^www\.", "", regex=True)
)

    # save the cleaned dataframe
    os.makedirs(os.path.dirname(FINAL_OUTPUT_FILE) or ".", exist_ok=True)
    balanced_df.to_csv(FINAL_OUTPUT_FILE, index=False)
    
    print(f"\nFinal dataset saved to: {FINAL_OUTPUT_FILE}")
    print("\nFinal Class Distribution:")
    print(f"\n{balanced_df['label'].value_counts()}")

if __name__ == "__main__":
    main()