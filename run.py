"""Entry point for the Flask application.

Usage:
    python run.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.predictor import StackingPredictor
from src.app import create_app

models_dir = os.path.join(os.path.dirname(__file__), 'saved_models')

print("Loading models from:", models_dir)
predictor = StackingPredictor(models_dir)
print("Models loaded successfully!")

app = create_app(predictor=predictor)
app.run(debug=True, host='127.0.0.1', port=5000)
