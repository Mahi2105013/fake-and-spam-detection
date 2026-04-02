import numpy as np
import pandas as pd

META_DIM = 15

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
    meta['has_promo'] = df['text'].str.contains(
        r'subscribe|follow us|check out|youtube|instagram|tiktok|'
        r'giveaway|unboxing|our channel|our website|our page',
        case=False, regex=True
    ).astype(int)
    meta['type_token_ratio']    = df['text'].apply(
        lambda x: len(set(x.lower().split())) / (len(x.split()) + 1))
    meta['rating'] = (df['rating'].fillna(3.0) - 3.0) / 2.0
    # One-hot encode the category to give it a vital role
    all_categories = ['Electronics', 'Home_and_Kitchen', 'Toys_and_Games', 
                      'Sports_and_Outdoors', 'Clothing_Shoes_and_Jewelry']
    for cat in all_categories:
        meta[f'category_{cat}'] = (df['category'] == cat).astype(float)
        
    # return meta.values.astype(np.float32)
    return meta.values.astype(np.float32)


def compute_rating_sample_weights(df):
    """Compute normalized sample weights from label and rating frequency.

    The final weight is a blend of:
    - label imbalance: keeps Real from dominating overall training
    - label/rating imbalance: keeps common rating buckets from overwhelming
      rare ones within each label

    The weights are normalized to mean 1.0 so they rebalance the loss without
    making training unstable.
    """
    tmp = df[['label', 'rating']].copy()
    tmp['rating'] = tmp['rating'].fillna(3.0).astype(float)

    total = len(tmp)
    label_counts = tmp['label'].value_counts().to_dict()
    combo_counts = tmp.groupby(['label', 'rating']).size().to_dict()

    num_labels = tmp['label'].nunique()
    num_combos = len(combo_counts)

    weights = []
    for label, rating in tmp.itertuples(index=False, name=None):
        label_weight = total / (num_labels * label_counts[label])
        combo_weight = total / (num_combos * combo_counts[(label, rating)])
        weights.append((label_weight * combo_weight) ** 0.5)

    weights = np.asarray(weights, dtype=np.float32)
    weights /= weights.mean()
    return weights


def get_preds_conf(probs):
    """Return (hard predictions, confidence = max probability) from a probability array."""
    preds = np.argmax(probs, axis=1)
    conf  = np.max(probs, axis=1)
    return preds, conf
