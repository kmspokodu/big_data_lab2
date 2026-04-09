import configparser
import os
import pickle
import traceback
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys

from logger import Logger

SHOW_LOG = True

app = Flask(__name__, static_folder='static')
CORS(app)

logger = Logger(SHOW_LOG)
log = logger.get_logger(__name__)

# Глобальные переменные для модели
classifier = None
model_path = None


def load_model():
    """Загружает обученную модель"""
    global classifier, model_path
    
    try:
        config = configparser.ConfigParser()
        config.read("config.ini")
        model_path = config["LOG_REG"]["path"]
        
        with open(model_path, 'rb') as f:
            classifier = pickle.load(f)
        log.info(f"Model loaded from {model_path}")
        return True
    except Exception as e:
        log.error(f"Failed to load model: {traceback.format_exc()}")
        return False


@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    if classifier is None:
        return jsonify({
            'status': 'unhealthy',
            'message': 'Model not loaded'
        }), 503
    
    return jsonify({
        'status': 'healthy',
        'model': 'LogisticRegression',
        'model_path': model_path
    }), 200


@app.route('/api/v1/predict', methods=['POST'])
def predict():
    """
    Предсказание для одного или нескольких образцов
    
    Expected JSON:
    {
        "features": [[feature1, feature2, ..., feature60], ...]
    }
    
    Response:
    {
        "predictions": ["M" or "R", ...],
        "probabilities": [[prob_M, prob_R], ...],
        "samples": N
    }
    """
    try:
        if classifier is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({'error': 'Missing "features" in request body'}), 400
        
        features = np.array(data['features'])
        
        # Проверка размерности
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        if features.shape[1] != 60:
            return jsonify({
                'error': f'Expected 60 features, got {features.shape[1]}'
            }), 400
        
        # Предсказание
        predictions = classifier.predict(features)
        probabilities = classifier.predict_proba(features)
        
        log.info(f"Made {len(predictions)} predictions")
        
        return jsonify({
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'classes': classifier.classes_.tolist(),
            'samples': len(predictions)
        }), 200
        
    except Exception as e:
        log.error(f"Prediction error: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/prompt', methods=['POST'])
def prompt():
    data = request.get_json()
    text = data.get("prompt", "")

    try:
        arr = [float(x.strip()) for x in text.split(',')]

        if len(arr) != 60:
            return jsonify({'error': 'Expected 60 numbers'}), 400

        features = np.array(arr).reshape(1, -1)

        prediction = classifier.predict(features)[0]
        prob = classifier.predict_proba(features)[0]

        return jsonify({
            'prediction': prediction,
            'probabilities': prob.tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/v1/predict/sonar', methods=['POST'])
def predict_sonar():
    """
    Специализированный endpoint для Sonar датасета
    
    Expected JSON:
    {
        "sonar_data": {
            "feature_0": value,
            "feature_1": value,
            ...,
            "feature_59": value
        }
    }
    """
    try:
        if classifier is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        data = request.get_json()
        
        if not data or 'sonar_data' not in data:
            return jsonify({'error': 'Missing "sonar_data" in request body'}), 400
        
        sonar_data = data['sonar_data']
        
        # Преобразуем в массив в правильном порядке
        features = []
        for i in range(60):
            key = f'feature_{i}'
            if key not in sonar_data:
                return jsonify({
                    'error': f'Missing {key}'
                }), 400
            features.append(sonar_data[key])
        
        features = np.array(features).reshape(1, -1)
        
        # Предсказание
        prediction = classifier.predict(features)[0]
        probability = classifier.predict_proba(features)[0]
        
        log.info(f"Sonar prediction: {prediction}")
        
        return jsonify({
            'prediction': prediction,
            'confidence': {
                'M': float(probability[0]),
                'R': float(probability[1])
            },
            'interpretation': 'M = Mine (мина), R = Rock (скала)'
        }), 200
        
    except Exception as e:
        log.error(f"Sonar prediction error: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/model-info', methods=['GET'])
def model_info():
    """Информация о модели"""
    if classifier is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    return jsonify({
        'model_type': 'LogisticRegression',
        'n_features': 60,
        'classes': classifier.classes_.tolist(),
        'dataset': 'Sonar',
        'input_format': 'Array of 60 features or dictionary with feature_0 to feature_59'
    }), 200


@app.route('/api/v1/batch-predict', methods=['POST'])
def batch_predict():
    """
    Batch предсказание для CSV данных
    
    Expected JSON или CSV в multipart/form-data
    """
    try:
        if classifier is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        if 'file' in request.files:
            file = request.files['file']
            df = pd.read_csv(file, header=None)
            features = df.values
        else:
            data = request.get_json()
            if not data or 'features' not in data:
                return jsonify({'error': 'Missing "features" in request body'}), 400
            features = np.array(data['features'])
        
        if features.shape[1] != 60:
            return jsonify({
                'error': f'Expected 60 features, got {features.shape[1]}'
            }), 400
        
        # Предсказание
        predictions = classifier.predict(features)
        probabilities = classifier.predict_proba(features)
        
        log.info(f"Batch prediction: {len(predictions)} samples")
        
        return jsonify({
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'classes': classifier.classes_.tolist(),
            'total_samples': len(predictions)
        }), 200
        
    except Exception as e:
        log.error(f"Batch prediction error: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500


# Debugging: Print all available routes
print(app.url_map)

if __name__ == '__main__':
    if not load_model():
        log.warning("Starting without model - predictions will fail until model is loaded")
    
    log.info("Starting Flask API on 0.0.0.0:55566")
    app.run(host='0.0.0.0', port=55566, debug=False)
