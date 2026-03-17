"""Integration tests for Flask endpoints."""

import json
import pytest


class TestIndexPage:

    def test_index_returns_200(self, client):
        response = client.get('/')
        assert response.status_code == 200

    def test_index_contains_form_elements(self, client):
        response = client.get('/')
        html = response.data.decode()
        assert '<textarea' in html
        assert 'rating' in html.lower()
        assert 'category' in html.lower()
        assert 'Electronics' in html


class TestPredictEndpoint:

    def test_valid_prediction(self, client, sample_review_dict):
        response = client.post('/predict',
                               data=json.dumps(sample_review_dict),
                               content_type='application/json')
        assert response.status_code == 200
        data = response.get_json()
        assert 'prediction' in data
        assert 'confidence' in data
        assert data['prediction'] in ['Real', 'Spam', 'Fake']

    def test_empty_text_returns_400(self, client):
        response = client.post('/predict',
                               data=json.dumps({'text': '', 'rating': 3.0,
                                                'category': 'Electronics'}),
                               content_type='application/json')
        assert response.status_code == 400
        assert 'error' in response.get_json()

    def test_missing_text_returns_400(self, client):
        response = client.post('/predict',
                               data=json.dumps({'rating': 3.0,
                                                'category': 'Electronics'}),
                               content_type='application/json')
        assert response.status_code == 400

    def test_invalid_rating_too_high(self, client):
        response = client.post('/predict',
                               data=json.dumps({'text': 'Some review',
                                                'rating': 6.0,
                                                'category': 'Electronics'}),
                               content_type='application/json')
        assert response.status_code == 400

    def test_invalid_rating_too_low(self, client):
        response = client.post('/predict',
                               data=json.dumps({'text': 'Some review',
                                                'rating': 0.0,
                                                'category': 'Electronics'}),
                               content_type='application/json')
        assert response.status_code == 400

    def test_invalid_category(self, client):
        response = client.post('/predict',
                               data=json.dumps({'text': 'Some review',
                                                'rating': 3.0,
                                                'category': 'InvalidCat'}),
                               content_type='application/json')
        assert response.status_code == 400

    def test_non_json_request_rejected(self, client):
        response = client.post('/predict',
                               data='not json',
                               content_type='text/plain')
        assert response.status_code in (400, 415)

    def test_predictor_called_with_correct_args(self, client, mock_predictor,
                                                 sample_review_dict):
        client.post('/predict',
                    data=json.dumps(sample_review_dict),
                    content_type='application/json')
        mock_predictor.predict.assert_called_once_with(
            sample_review_dict['text'],
            sample_review_dict['rating'],
            sample_review_dict['category'])
