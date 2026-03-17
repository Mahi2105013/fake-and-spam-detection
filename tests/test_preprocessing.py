"""Unit tests for extract_metadata and get_preds_conf."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import extract_metadata, get_preds_conf, META_DIM


class TestExtractMetadata:

    def test_output_shape_single_row(self, sample_review_df):
        result = extract_metadata(sample_review_df)
        assert result.shape == (1, META_DIM)
        assert result.dtype == np.float32

    def test_output_shape_multiple_rows(self):
        df = pd.DataFrame({
            'text': ['Hello world', 'Another review here', 'Third one'],
            'rating': [1.0, 3.0, 5.0],
        })
        result = extract_metadata(df)
        assert result.shape == (3, META_DIM)

    def test_review_length_feature(self):
        text = 'Hello world'
        df = pd.DataFrame({'text': [text], 'rating': [3.0]})
        result = extract_metadata(df)
        assert result[0, 0] == len(text)

    def test_word_count_feature(self):
        df = pd.DataFrame({'text': ['one two three four'], 'rating': [3.0]})
        result = extract_metadata(df)
        assert result[0, 1] == 4

    def test_exclamation_count_feature(self):
        df = pd.DataFrame({'text': ['Great!! Amazing!'], 'rating': [5.0]})
        result = extract_metadata(df)
        assert result[0, 2] == 3

    def test_has_url_positive(self):
        df = pd.DataFrame({'text': ['visit http://example.com'], 'rating': [3.0]})
        assert extract_metadata(df)[0, 4] == 1

    def test_has_url_negative(self):
        df = pd.DataFrame({'text': ['no link here'], 'rating': [3.0]})
        assert extract_metadata(df)[0, 4] == 0

    def test_rating_preserved(self):
        df = pd.DataFrame({'text': ['test text'], 'rating': [4.5]})
        result = extract_metadata(df)
        assert result[0, 8] == pytest.approx(4.5)

    def test_nan_rating_defaults_to_3(self):
        df = pd.DataFrame({'text': ['test text'], 'rating': [float('nan')]})
        result = extract_metadata(df)
        assert result[0, 8] == 3.0

    def test_all_features_finite(self):
        df = pd.DataFrame({
            'text': ['A normal product review with some words.'],
            'rating': [4.0],
        })
        result = extract_metadata(df)
        assert np.all(np.isfinite(result))


class TestGetPredsConf:

    def test_basic_prediction(self):
        probs = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])
        preds, conf = get_preds_conf(probs)
        assert list(preds) == [1, 0]
        assert list(conf) == pytest.approx([0.7, 0.8])

    def test_single_row(self):
        probs = np.array([[0.3, 0.3, 0.4]])
        preds, conf = get_preds_conf(probs)
        assert preds[0] == 2
        assert conf[0] == pytest.approx(0.4)
