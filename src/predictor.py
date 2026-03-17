import os
import numpy as np
import pandas as pd
import torch
import joblib
from scipy.sparse import hstack as sparse_hstack
from transformers import RobertaTokenizer

from .models import GatedFusionModel, TextSCNN, META_DIM
from .preprocessing import extract_metadata, get_preds_conf, CLASS_NAMES

VOCAB_SIZE = 50265   # roberta-base tokenizer vocab size
MAX_LENGTH = 128     # same as notebook training


class StackingPredictor:
    """Loads 3 base models + meta-learner + TF-IDF vectorizer from disk.

    Provides .predict(text, rating, category) that returns:
        {'prediction': 'Real'/'Spam'/'Fake', 'confidence': float, 'probabilities': dict}
    """

    def __init__(self, models_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_dir = models_dir
        self._load_all()

    def _load_all(self):
        d = self.models_dir

        # 1. RoBERTa tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(os.path.join(d, 'tokenizer'))

        # 2. GatedFusionModel
        self.gated_fusion = GatedFusionModel(
            roberta_name='roberta-base', num_labels=3,
            meta_dim=META_DIM, meta_hidden=32, dropout=0.3)
        self.gated_fusion.load_state_dict(
            torch.load(os.path.join(d, 'gated_fusion.pt'),
                       map_location=self.device, weights_only=True),
            strict=False)
        self.gated_fusion.to(self.device)
        self.gated_fusion.eval()

        # 3. TextSCNN
        self.textscnn = TextSCNN(
            vocab_size=VOCAB_SIZE, embed_dim=128, num_labels=3,
            sentence_per_review=8, words_per_sentence=16)
        self.textscnn.load_state_dict(
            torch.load(os.path.join(d, 'textscnn.pt'),
                       map_location=self.device, weights_only=True),
            strict=False)
        self.textscnn.to(self.device)
        self.textscnn.eval()

        # 4. XGBoost classifier
        self.xgb_model = joblib.load(os.path.join(d, 'xgboost.pkl'))

        # 5. TF-IDF vectorizer (must be the same one used during training)
        self.tfidf = joblib.load(os.path.join(d, 'tfidf_vectorizer.pkl'))

        # 6. Meta-learner (LogisticRegression)
        self.meta_learner = joblib.load(os.path.join(d, 'meta_learner.pkl'))

    def predict(self, text, rating, category):
        """Run the full stacking ensemble on a single review.

        Args:
            text: review text string
            rating: float 1.0-5.0
            category: one of the 5 category strings (not used by models,
                      included for UI completeness)
        Returns:
            dict with 'prediction', 'confidence', and 'probabilities'
        """
        # Build single-row DataFrame for extract_metadata
        df = pd.DataFrame({
            'text': [text],
            'rating': [float(rating)],
            'category': [category],
        })
        meta = extract_metadata(df)  # shape (1, 9)

        # Tokenize for PyTorch models
        encoding = self.tokenizer(
            text, add_special_tokens=True, max_length=MAX_LENGTH,
            padding='max_length', truncation=True,
            return_attention_mask=True, return_tensors='pt')
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        metadata_tensor = torch.tensor(meta, dtype=torch.float32).to(self.device)

        # Base model 1: GatedFusionModel
        with torch.no_grad():
            gf_out = self.gated_fusion(
                input_ids=input_ids,
                attention_mask=attention_mask,
                metadata=metadata_tensor)
            gf_probs = torch.softmax(gf_out.logits, dim=1).cpu().numpy()

        # Base model 2: TextSCNN
        with torch.no_grad():
            scnn_out = self.textscnn(input_ids=input_ids)
            scnn_probs = torch.softmax(scnn_out.logits, dim=1).cpu().numpy()

        # Base model 3: XGBoost
        tfidf_vec = self.tfidf.transform([text])
        xgb_features = sparse_hstack([tfidf_vec, meta])
        xgb_probs = self.xgb_model.predict_proba(xgb_features)

        # Stack (1, 9) and feed to meta-learner
        stacked = np.hstack([gf_probs, scnn_probs, xgb_probs])
        final_probs = self.meta_learner.predict_proba(stacked)

        pred_idx = int(np.argmax(final_probs, axis=1)[0])
        confidence = float(np.max(final_probs, axis=1)[0])

        return {
            'prediction': CLASS_NAMES[pred_idx],
            'confidence': round(confidence * 100, 1),
            'probabilities': {
                name: round(float(p) * 100, 1)
                for name, p in zip(CLASS_NAMES, final_probs[0])
            },
        }
