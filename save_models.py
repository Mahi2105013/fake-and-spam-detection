"""Save all trained model artifacts after notebook training.

This script is meant to be run AFTER the notebook
(notebooks/fake-and-spam-reviews-detection.ipynb) has finished training.

You can either:
  1. Copy the save_artifacts() code below into a new notebook cell after training
  2. Run this script if model variables are accessible (e.g., via %run in Jupyter)

Artifacts saved to saved_models/:
  - gated_fusion.pt       (GatedFusionModel state_dict)
  - textscnn.pt           (TextSCNN state_dict)
  - xgboost.pkl           (XGBClassifier via joblib)
  - tfidf_vectorizer.pkl  (TfidfVectorizer via joblib)
  - meta_learner.pkl      (LogisticRegression meta-learner via joblib)
  - tokenizer/            (RoBERTa tokenizer files)
"""

import os
import torch
import joblib


def save_artifacts(roberta_model, scnn_model, xgb_model,
                   tfidf, meta_learner, roberta_tokenizer,
                   save_dir=None):
    """Save all model artifacts to disk.

    Args:
        roberta_model: trained GatedFusionModel
        scnn_model: trained TextSCNN
        xgb_model: trained XGBClassifier
        tfidf: fitted TfidfVectorizer
        meta_learner: fitted LogisticRegression
        roberta_tokenizer: RobertaTokenizer
        save_dir: directory to save to (default: saved_models/ next to this script)
    """
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'tokenizer'), exist_ok=True)

    # PyTorch models (state_dict only)
    torch.save(roberta_model.state_dict(), os.path.join(save_dir, 'gated_fusion.pt'))
    print("Saved gated_fusion.pt")

    torch.save(scnn_model.state_dict(), os.path.join(save_dir, 'textscnn.pt'))
    print("Saved textscnn.pt")

    # XGBoost model
    joblib.dump(xgb_model, os.path.join(save_dir, 'xgboost.pkl'))
    print("Saved xgboost.pkl")

    # TF-IDF vectorizer
    joblib.dump(tfidf, os.path.join(save_dir, 'tfidf_vectorizer.pkl'))
    print("Saved tfidf_vectorizer.pkl")

    # Meta-learner
    joblib.dump(meta_learner, os.path.join(save_dir, 'meta_learner.pkl'))
    print("Saved meta_learner.pkl")

    # Tokenizer
    roberta_tokenizer.save_pretrained(os.path.join(save_dir, 'tokenizer'))
    print("Saved tokenizer/")

    print(f"\nAll artifacts saved to {save_dir}")


# ── Usage from notebook ──────────────────────────────────────────────
# After all training cells have run, add a new cell with:
#
#   import sys, os
#   sys.path.insert(0, '/home/mezba/Downloads/CSE 330/fake review')
#   from save_models import save_artifacts
#
#   save_artifacts(
#       roberta_model=roberta_model,
#       scnn_model=scnn_model,
#       xgb_model=xgb_model,
#       tfidf=tfidf,
#       meta_learner=meta_learner,
#       roberta_tokenizer=roberta_tokenizer,
#   )
