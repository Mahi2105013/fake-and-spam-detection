import os
import sys
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def sample_review_df():
    """Single-row DataFrame matching extract_metadata's expected input."""
    return pd.DataFrame({
        'text': ['This product is absolutely amazing! Best purchase I have ever made!!!'],
        'rating': [5.0],
        'category': ['Electronics'],
    })


@pytest.fixture
def sample_review_dict():
    """Dict matching the JSON body for the /predict endpoint."""
    return {
        'text': 'This product is absolutely amazing! Best purchase I have ever made!!!',
        'rating': 5.0,
        'category': 'Electronics',
    }


@pytest.fixture
def mock_predictor():
    """MagicMock that mimics StackingPredictor.predict() return value."""
    predictor = MagicMock()
    predictor.predict.return_value = {
        'prediction': 'Spam',
        'confidence': 87.5,
        'probabilities': {'Real': 5.2, 'Spam': 87.5, 'Fake': 7.3},
    }
    return predictor


@pytest.fixture
def client(mock_predictor):
    """Flask test client with a mocked predictor (no real models loaded)."""
    from src.app import create_app
    app = create_app(predictor=mock_predictor)
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c
