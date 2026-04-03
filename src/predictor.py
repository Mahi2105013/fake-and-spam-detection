import os
import json
import numpy as np
import pandas as pd
import torch
import joblib
from scipy.sparse import hstack as sparse_hstack
from transformers import RobertaTokenizer

try:
    import shap
except ImportError:  # pragma: no cover - optional dependency
    shap = None

from .models import GatedFusionModel, TextSCNN, META_DIM
from .preprocessing import extract_metadata, get_preds_conf, CLASS_NAMES

VOCAB_SIZE = 50265   # roberta-base tokenizer vocab size
MAX_LENGTH = 128     # same as notebook training
META_FEATURE_NAMES = [
   "review_length",           # 0
    "word_count",              # 1
    "exclamation_count",       # 2
    "all_caps_ratio",          # 3  ← NEW: only SHOUTING caps like BUY NOW
    "sentence_caps_ratio",     # 4  ← NEW: normal sentence/pronoun caps
    "has_url",                 # 5
    "avg_word_length",         # 6
    "punctuation_density",     # 7
    "type_token_ratio",        # 8
    "rating_norm",             # 9
    "rating_sentiment_mismatch", # 10 ← NEW
    "has_promo",               # 11
    "category_Electronics",    # 12
    "category_Home_and_Kitchen", # 13
    "category_Toys_and_Games", # 14
    "category_Sports_and_Outdoors", # 15
    "category_Clothing_Shoes_and_Jewelry", # 16
]


class StackingPredictor:
    """Loads 3 base models + meta-learner + TF-IDF vectorizer from disk.

    Provides .predict(text, rating, category) that returns:
        {'prediction': 'Real'/'Spam'/'Fake', 'confidence': float, 'probabilities': dict}
    """

    def __init__(self, models_dir):
        # For web service: load models on CPU by default.
        # This avoids GPU OOM issues on shared/constrained infrastructure.
        # Inference speed on CPU is acceptable for most reviews.
        self.device = torch.device('cpu')
        self.models_dir = self._resolve_models_dir(models_dir)
        self._load_all()

    @staticmethod
    def _resolve_models_dir(models_dir):
        """Resolve model artifacts directory from either flat or nested layout."""
        required_files = {
            'gated_fusion.pt',
            'textscnn.pt',
            'xgboost.pkl',
            'tfidf_vectorizer.pkl',
            'meta_learner.pkl',
        }

        direct_tokenizer = os.path.join(models_dir, 'tokenizer')
        direct_ok = os.path.isdir(direct_tokenizer) and all(
            os.path.isfile(os.path.join(models_dir, name)) for name in required_files)
        if direct_ok:
            return models_dir

        nested_dir = os.path.join(models_dir, 'saved_models')
        nested_tokenizer = os.path.join(nested_dir, 'tokenizer')
        nested_ok = os.path.isdir(nested_tokenizer) and all(
            os.path.isfile(os.path.join(nested_dir, name)) for name in required_files)
        if nested_ok:
            return nested_dir

        raise FileNotFoundError(
            f"Could not find model artifacts in '{models_dir}'. "
            "Expected either a flat layout with tokenizer/ and model files, "
            "or a nested saved_models/ directory containing them.")

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

        # 7. Optional dataset-level evaluation metrics for UI display
        self.model_metrics = None
        metrics_path = os.path.join(d, 'metrics.json')
        if os.path.isfile(metrics_path):
            try:
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    parsed = json.load(f)
                if isinstance(parsed, dict):
                    self.model_metrics = parsed
            except Exception:
                self.model_metrics = None

        # 8. Optional SHAP explainer for metadata attribution on XGBoost branch
        self._shap_explainer = None
        self._shap_error = None
        if shap is None:
            self._shap_error = 'SHAP is not installed. Run: pip install shap'
        else:
            try:
                self._shap_explainer = shap.TreeExplainer(self.xgb_model)
            except Exception as exc:  # pragma: no cover - depends on model binary
                self._shap_error = f'Could not initialize SHAP explainer: {exc}'

    @staticmethod
    def _clean_token(token):
        token = token.replace('Ġ', '').replace('Ċ', '').strip()
        return token

    def _explain_metadata_shap(self, xgb_features, meta_values, pred_idx, top_k=6):
        if self._shap_explainer is None:
            return {
                'available': False,
                'reason': self._shap_error or 'SHAP explainer unavailable',
                'top_features': [],
            }

        try:
            dense_features = xgb_features.toarray()
            shap_values = self._shap_explainer.shap_values(dense_features)

            # SHAP output format differs between versions/models.
            if isinstance(shap_values, list):
                class_values = np.asarray(shap_values[pred_idx][0])
            else:
                arr = np.asarray(shap_values)
                if arr.ndim == 3:
                    class_values = arr[0, :, pred_idx]
                elif arr.ndim == 2:
                    class_values = arr[0]
                else:
                    raise ValueError('Unsupported SHAP output shape.')

            tfidf_dim = dense_features.shape[1] - len(META_FEATURE_NAMES)
            meta_shap = class_values[tfidf_dim:tfidf_dim + len(META_FEATURE_NAMES)]

            abs_sum = float(np.sum(np.abs(meta_shap))) + 1e-12
            rows = []
            for i, feature in enumerate(META_FEATURE_NAMES):
                shap_val = float(meta_shap[i])
                rows.append({
                    'feature': feature,
                    'value': round(float(meta_values[i]), 4),
                    'shap_impact': round(shap_val, 6),
                    'impact_percent': round(abs(shap_val) / abs_sum * 100.0, 2),
                    'direction': 'pushes_toward_prediction' if shap_val >= 0 else 'pushes_away_from_prediction',
                })

            rows.sort(key=lambda x: abs(x['shap_impact']), reverse=True)
            return {
                'available': True,
                'top_features': rows[:top_k],
            }
        except Exception as exc:  # pragma: no cover - runtime explainability fallback
            return {
                'available': False,
                'reason': f'Failed to compute SHAP values: {exc}',
                'top_features': [],
            }

    def _explain_tokens_attention(self, input_ids, attention_mask, pred_idx,
                                  include_bertviz=False, top_k=10):
        try:
            with torch.no_grad():
                out = self.gated_fusion.roberta(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    return_dict=True,
                )

            last_attn = out.attentions[-1][0]  # [heads, seq_len, seq_len]
            cls_to_token = last_attn[:, 0, :].mean(dim=0).cpu().numpy()
            ids = input_ids[0].detach().cpu().tolist()
            mask = attention_mask[0].detach().cpu().numpy().astype(bool)
            raw_tokens = self.tokenizer.convert_ids_to_tokens(ids)

            token_rows = []
            filtered_tokens = []
            filtered_indices = []
            for idx, (tok, keep) in enumerate(zip(raw_tokens, mask)):
                if not keep:
                    continue
                if tok in {'<s>', '</s>', '<pad>'}:
                    continue
                clean = self._clean_token(tok)
                if not clean:
                    continue
                score = float(cls_to_token[idx])
                token_rows.append({
                    'token': clean,
                    'attention_score': round(score, 6),
                })
                filtered_tokens.append(clean)
                filtered_indices.append(idx)

            token_rows.sort(key=lambda x: x['attention_score'], reverse=True)

            bertviz_payload = None
            if include_bertviz:
                # Build a compact BERTViz payload using the same token subset shown in UI.
                compact_attn = last_attn[:, filtered_indices, :][:, :, filtered_indices]
                bertviz_payload = {
                    'predicted_class': CLASS_NAMES[pred_idx],
                    'tokens': filtered_tokens,
                    'attentions': [compact_attn.detach().cpu().tolist()],  # [layers=1, heads, seq, seq]
                    'note': 'Pass attentions and tokens to bertviz.head_view in notebook for interactive inspection.',
                }

            return {
                'available': True,
                'top_tokens': token_rows[:top_k],
                'bertviz': bertviz_payload,
            }
        except Exception as exc:  # pragma: no cover - runtime explainability fallback
            return {
                'available': False,
                'reason': f'Failed to compute token influence: {exc}',
                'top_tokens': [],
            }

    def predict(self, text, rating, category, include_bertviz=False):
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

        
        XGB_DROP = {6,9}
        XGB_META_COLS = [i for i in range(META_DIM) if i not in XGB_DROP]
        tfidf_vec = self.tfidf.transform([text])
        xgb_features = sparse_hstack([tfidf_vec, meta[:, XGB_META_COLS]])
        xgb_probs = self.xgb_model.predict_proba(xgb_features)

        stacked = np.hstack([gf_probs, scnn_probs, xgb_probs])
        final_probs = self.meta_learner.predict_proba(stacked)

        pred_idx = int(np.argmax(final_probs, axis=1)[0])
        confidence = float(np.max(final_probs, axis=1)[0])
        meta_explanation = self._explain_metadata_shap(
            xgb_features=xgb_features,
            meta_values=meta[0],
            pred_idx=pred_idx,
        )
        token_explanation = self._explain_tokens_attention(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pred_idx=pred_idx,
            include_bertviz=include_bertviz,
        )

        return {
            'prediction': CLASS_NAMES[pred_idx],
            'confidence': round(confidence * 100, 1),
            'probabilities': {
                name: round(float(p) * 100, 1)
                for name, p in zip(CLASS_NAMES, final_probs[0])
            },
            'model_metrics': self.model_metrics,
            'explanations': {
                'metadata_shap': meta_explanation,
                'token_attention': token_explanation,
            },
        }
