"""Unit tests for predictor output format and contract."""

import os
import pytest

from src.preprocessing import CLASS_NAMES


class TestPredictorOutput:

    def test_returns_required_keys(self, mock_predictor, sample_review_dict):
        result = mock_predictor.predict(
            sample_review_dict['text'],
            sample_review_dict['rating'],
            sample_review_dict['category'])
        assert 'prediction' in result
        assert 'confidence' in result
        assert 'probabilities' in result

    def test_prediction_is_valid_label(self, mock_predictor, sample_review_dict):
        result = mock_predictor.predict(
            sample_review_dict['text'],
            sample_review_dict['rating'],
            sample_review_dict['category'])
        assert result['prediction'] in CLASS_NAMES

    def test_confidence_in_range(self, mock_predictor, sample_review_dict):
        result = mock_predictor.predict(
            sample_review_dict['text'],
            sample_review_dict['rating'],
            sample_review_dict['category'])
        assert 0 <= result['confidence'] <= 100

    def test_probabilities_sum_roughly_100(self, mock_predictor, sample_review_dict):
        result = mock_predictor.predict(
            sample_review_dict['text'],
            sample_review_dict['rating'],
            sample_review_dict['category'])
        total = sum(result['probabilities'].values())
        assert total == pytest.approx(100, abs=1.0)

    def test_probabilities_has_all_classes(self, mock_predictor, sample_review_dict):
        result = mock_predictor.predict(
            sample_review_dict['text'],
            sample_review_dict['rating'],
            sample_review_dict['category'])
        for name in CLASS_NAMES:
            assert name in result['probabilities']


# Conditionally run a real prediction if models are on disk
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')


@pytest.mark.skipif(
    not os.path.isdir(MODELS_DIR)
    or not os.path.exists(os.path.join(MODELS_DIR, 'meta_learner.pkl')),
    reason="Trained models not found in saved_models/")
class TestRealPredictor:

    def test_smoke_prediction(self):
        from src.predictor import StackingPredictor
        predictor = StackingPredictor(MODELS_DIR)
        result = predictor.predict(
            "This is a great product, I love it!", 5.0, "Electronics")
        assert result['prediction'] in CLASS_NAMES
        assert 0 <= result['confidence'] <= 100
