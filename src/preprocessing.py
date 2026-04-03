import numpy as np
import pandas as pd
import re as _re
from textblob import TextBlob

META_DIM = 18

CATEGORIES = [
    'Home_and_Kitchen',
    'Electronics',
    'Sports_and_Outdoors',
    'Clothing_Shoes_and_Jewelry',
    'Toys_and_Games',
]

CLASS_NAMES = ['Real', 'Spam', 'Fake']


def extract_metadata(df):
  
    meta = pd.DataFrame()

    meta['review_length']     = df['text'].str.len()
    meta['word_count']        = df['text'].str.split().str.len()
    meta['exclamation_count'] = df['text'].str.count('!')

    def all_caps_ratio(text):
        
        words = text.split()
        if not words: return 0.0
        all_caps_chars = sum(len(w) for w in words
                             if w.isalpha() and w.isupper() and len(w) > 1)
        return all_caps_chars / (len(text) + 1)

    def sentence_caps_ratio(text):
        upper_total = sum(1 for c in text if c.isupper())
        words = text.split()
        all_caps_chars = sum(len(w) for w in words
                             if w.isalpha() and w.isupper() and len(w) > 1)
        normal_caps = upper_total - all_caps_chars
        return normal_caps / (len(text) + 1)

    meta['all_caps_ratio']      = df['text'].apply(all_caps_ratio)
    meta['sentence_caps_ratio'] = df['text'].apply(sentence_caps_ratio)

    meta['has_url']             = df['text'].str.contains(
                                    r'https|http|www|\.com', regex=True).astype(int)
    meta['avg_word_length']     = df['text'].apply(
                                    lambda x: np.mean([len(w) for w in x.split()]+[0]))
    meta['punctuation_density'] = df['text'].apply(
                                    lambda x: sum(1 for c in x if c in '.,!?;:')/(len(x)+1))
    meta['type_token_ratio']    = df['text'].apply(
                                    lambda x: len(set(x.lower().split()))/(len(x.split())+1))

    rating_norm = (df['rating'].fillna(3.0).astype(float) - 3.0) / 2.0
    meta['rating_norm'] = rating_norm

    sentiment = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    meta['rating_sentiment_mismatch'] = (rating_norm - sentiment).abs()

    meta['has_promo'] = df['text'].str.contains(
        r'subscribe|follow us|check out|youtube|instagram|tiktok|snapchat|'
        r'giveaway|unboxing|our channel|our website|our page|my channel|'
        r'use code|promo code|discount code|affiliate|sponsored',
        case=False, regex=True
    ).astype(int)

    for cat in ['Electronics','Home_and_Kitchen','Toys_and_Games',
                'Sports_and_Outdoors','Clothing_Shoes_and_Jewelry']:
        meta[f'category_{cat}'] = (df['category'] == cat).astype(float)

    return meta.values.astype(np.float32)

def compute_rating_sample_weights(df):
    
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
    preds = np.argmax(probs, axis=1)
    conf  = np.max(probs, axis=1)
    return preds, conf
