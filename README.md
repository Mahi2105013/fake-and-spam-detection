# Fake and Spam Reviews Detection

A machine learning project that detects fake, spam, and genuine product reviews using an ensemble of deep learning and traditional ML models. This project combines RoBERTa transformers, CNNs, XGBoost, and metadata features for robust multi-class classification.

## 📋 Overview

This system classifies Amazon product reviews into three categories:
- **Real** (Genuine reviews)
- **Spam** (Non-review content, off-topic)
- **Fake** (Intentionally deceptive or manipulated reviews)

The project uses a **stacking ensemble approach** combining:
- **RoBERTa + Gated Fusion** - Transformer-based text encoding with learned metadata fusion
- **TextSCNN** - Hierarchical CNN architecture for multi-scale text feature extraction
- **XGBoost** - Gradient boosting on TF-IDF + metadata features
- **Meta-Learner** - Logistic regression stacking layer

## 🎯 Key Features

### Advanced Metadata Extraction
Extracts 17 linguistic and behavioral features:
- Review structure: length, word count, punctuation density
- Language patterns: capitalization ratios, sentiment analysis, type-token ratio
- Category indicators and promotional content detection
- Rating-sentiment mismatch detection

### Ensemble Architecture
- **Gated Fusion**: Learns adaptive weighting between text embeddings and metadata
- **Stacking**: Combines predictions from all three models via a meta-learner
- **Sample Weighting**: Balances training across label and rating distributions

### Web Interface
Flask-based web application for real-time predictions with:
- Interactive UI for review submissions
- Per-review confidence scores
- Category selection support
- Visualization of model attention patterns

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- PyTorch 2.0+
- GPU recommended (CUDA 11.8+)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Mezba8053/fake-and-spam-detection.git
cd fake-and-spam-detection
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download pre-trained models** (optional)
```bash
# Models are already saved in saved_models/ directory
# If needed, retrain using: python notebooks/fake-spam-final.ipynb
```

### Running the Application

```bash
python run.py
```

Then visit: **http://127.0.0.1:5000**

## 📊 Model Performance

Ensemble results on test set:

| Model | Accuracy | Macro F1 | Real F1 | Spam F1 | Fake F1 |
|-------|----------|----------|---------|---------|---------|
| RoBERTa + GatedFusion | ~92% | ~90% | ~94% | ~87% | ~89% |
| TextSCNN | ~89% | ~87% | ~91% | ~84% | ~86% |
| XGBoost | ~88% | ~85% | ~89% | ~82% | ~84% |
| **Stacking Ensemble** | **~93%** | **~92%** | **~95%** | **~90%** | **~91%** |

## 📁 Project Structure

```
fake-and-spam-detection/
├── notebooks/
│   └── fake-spam-final.ipynb          # Training & evaluation notebook
├── src/
│   ├── app.py                         # Flask application
│   ├── models.py                      # Model architectures
│   ├── predictor.py                   # Prediction pipeline
│   ├── preprocessing.py               # Data preprocessing
│   └── templates/
│       └── index.html                 # Web UI
├── tests/
│   ├── test_app.py
│   ├── test_predictor.py
│   └── test_preprocessing.py
├── saved_models/
│   ├── gated_fusion.pt                # Trained model weights
│   ├── textscnn.pt
│   ├── tokenizer/                     # RoBERTa tokenizer
│   └── model metadata
├── data/
│   └── merged_fake_spam_reviews_augmented_final.csv
├── run.py                             # Application entry point
├── save_models.py                     # Model serialization
└── requirements.txt
```

## 🔧 Usage

### Web Interface
1. Enter a review text
2. Select product category (Electronics, Home & Kitchen, Toys & Games, etc.)
3. Provide rating (1-5 stars)
4. Click "Predict"
5. View classification result with confidence scores for each class

### API Usage (Python)
```python
from src.predictor import StackingPredictor
import os

models_dir = 'saved_models'
predictor = StackingPredictor(models_dir)

review = {
    'text': 'Great product! Shipped fast and excellent quality.',
    'rating': 5.0,
    'category': 'Electronics'
}

result = predictor.predict(review)
print(result['prediction'])        # 'Real', 'Spam', or 'Fake'
print(result['confidence_scores']) # {'Real': 0.95, 'Spam': 0.03, 'Fake': 0.02}
```

## 🎓 Training

To retrain models from scratch:

1. Open the Jupyter notebook:
```bash
jupyter notebook notebooks/fake-spam-final.ipynb
```

2. The notebook includes:
   - Data loading and preprocessing
   - Metadata feature extraction
   - Model training for all three architectures
   - Ensemble stacking and evaluation
   - Visualization of results

3. Save models:
```bash
python save_models.py
```

## 📚 Dataset

- **Source**: Amazon reviews (multiple categories)
- **Size**: ~25,000 annotated reviews
- **Labels**: Real (genuine), Spam (off-topic), Fake (deceptive)
- **Features**: Review text, rating, product category

Dataset preprocessing includes:
- Duplicate removal
- Conflicting label resolution
- Class balancing via sample weighting
- Train/val/test split (70/12.5/17.5)

## 🛠️ Technologies

- **Deep Learning**: PyTorch, Transformers (RoBERTa)
- **ML**: scikit-learn, XGBoost
- **NLP**: NLTK, TextBlob, TfidfVectorizer
- **Web**: Flask
- **Data**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

## 📄 Metadata Features

The system extracts these features for each review:

1. Review length & word count
2. Exclamation marks and punctuation density
3. Capitalization patterns (SHOUTING vs sentence case)
4. URL/hyperlink detection
5. Average word length
6. Type-token ratio (vocabulary richness)
7. Sentiment-rating mismatch
8. Promotional content indicators
9. Product category embeddings

## ✅ Testing

Run the test suite:
```bash
pytest tests/
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.



## 🔗 References

- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
- [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

---

**Last Updated**: April 2026
