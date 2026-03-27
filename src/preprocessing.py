import numpy as np
import pandas as pd

META_DIM = 14

CATEGORIES = [
    'Home_and_Kitchen',
    'Electronics',
    'Sports_and_Outdoors',
    'Clothing_Shoes_and_Jewelry',
    'Toys_and_Games',
]

CLASS_NAMES = ['Real', 'Spam', 'Fake']


def extract_metadata(df):
    """Extract 9 numeric features from a DataFrame with 'text' and 'rating' columns.

    Returns ndarray of shape (N, 9) with dtype float32.
    Features: review_length, word_count, exclamation_count, uppercase_ratio,
              has_url, avg_word_length, punctuation_density, type_token_ratio, rating
    """
    meta = pd.DataFrame()
    meta['review_length']       = df['text'].str.len()
    meta['word_count']          = df['text'].str.split().str.len()
    meta['exclamation_count']   = df['text'].str.count('!')
    meta['uppercase_ratio']     = df['text'].apply(
        lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1))
    meta['has_url']             = df['text'].str.contains(
        r'http|www|\.com', regex=True).astype(int)
    meta['avg_word_length']     = df['text'].apply(
        lambda x: np.mean([len(w) for w in x.split()] + [0]))
    meta['punctuation_density'] = df['text'].apply(
        lambda x: sum(1 for c in x if c in '.,!?;:') / (len(x) + 1))
    meta['type_token_ratio']    = df['text'].apply(
        lambda x: len(set(x.lower().split())) / (len(x.split()) + 1))
    meta['rating'] = df['rating'].fillna(3.0).astype(float)
    # One-hot encode the category to give it a vital role
    all_categories = ['Electronics', 'Home_and_Kitchen', 'Toys_and_Games', 
                      'Sports_and_Outdoors', 'Clothing_Shoes_and_Jewelry']
    for cat in all_categories:
        meta[f'category_{cat}'] = (df['category'] == cat).astype(float)
        
    # return meta.values.astype(np.float32)
    return meta.values.astype(np.float32)


def get_preds_conf(probs):
    """Return (hard predictions, confidence = max probability) from a probability array."""
    preds = np.argmax(probs, axis=1)
    conf  = np.max(probs, axis=1)
    return preds, conf
