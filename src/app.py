from flask import Flask, render_template, request, jsonify

from .preprocessing import CATEGORIES


def create_app(predictor=None):
    """Flask application factory.

    Args:
        predictor: a StackingPredictor instance (or mock for testing).
                   If None, loads real models from saved_models/ directory.
    """
    app = Flask(__name__)

    if predictor is None:
        import os
        from .predictor import StackingPredictor
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
        predictor = StackingPredictor(models_dir)

    app.config['PREDICTOR'] = predictor

    @app.route('/')
    def index():
        return render_template('index.html', categories=CATEGORIES)

    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request must be JSON'}), 400

        text = data.get('text', '').strip()
        rating = data.get('rating', 3.0)
        category = data.get('category', 'Electronics')
        include_bertviz = bool(data.get('include_bertviz', False))

        if not text:
            return jsonify({'error': 'Review text is required'}), 400
        try:
            rating = float(rating)
        except (TypeError, ValueError):
            return jsonify({'error': 'Rating must be a number'}), 400
        if rating < 1.0 or rating > 5.0:
            return jsonify({'error': 'Rating must be between 1 and 5'}), 400
        if category not in CATEGORIES:
            return jsonify({'error': 'Invalid category'}), 400

        if include_bertviz:
            result = app.config['PREDICTOR'].predict(
                text, rating, category, include_bertviz=True)
        else:
            result = app.config['PREDICTOR'].predict(text, rating, category)
        return jsonify(result)

    return app
