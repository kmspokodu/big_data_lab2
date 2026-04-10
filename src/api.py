import os
import pickle
import traceback
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime
import sys
from dotenv import load_dotenv

from logger import Logger
from config import config as app_config

# Load environment variables from .env file
load_dotenv()

SHOW_LOG = True

app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)

logger = Logger(SHOW_LOG)
log = logger.get_logger(__name__)

# Глобальные переменные для модели и MongoDB
classifier = None
model_path = None
mongodb_collection = None


def load_model():
    """Загружает обученную модель"""
    global classifier, model_path
    
    try:
        model_path = app_config.model_path
        
        with open(model_path, 'rb') as f:
            classifier = pickle.load(f)
        log.info(f"Model loaded from {model_path}")
        return True
    except Exception as e:
        log.error(f"Failed to load model: {traceback.format_exc()}")
        return False


def connect_mongodb():
    """Подключается к MongoDB с использованием конфигурации из переменных окружения"""
    global mongodb_collection
    
    try:
        mongo_config = app_config.mongodb
        
        
        client = MongoClient(mongo_config.connection_string, serverSelectionTimeoutMS=5000)
        
        client.admin.command('ping')
        
        db = client[mongo_config.database]
        mongodb_collection = db[mongo_config.collection]
        
        log.info(f"Connected to MongoDB: {mongo_config.uri_safe}/{mongo_config.collection}")
        return True
    except Exception as e:
        log.warning(f"Failed to connect to MongoDB: {traceback.format_exc()}")
        mongodb_collection = None
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
        
        # Подготовка ответа
        response_data = {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'classes': classifier.classes_.tolist(),
            'samples': len(predictions)
        }
        
        # Сохранение в MongoDB (если доступна БД)
        if mongodb_collection is not None:
            try:
                db_record = {
                    'type': 'predict',
                    'input_features': features.tolist(),
                    'predictions': predictions.tolist(),
                    'probabilities': probabilities.tolist(),
                    'timestamp': datetime.utcnow()
                }
                result = mongodb_collection.insert_one(db_record)
                response_data['db_id'] = str(result.inserted_id)
                log.info(f"Prediction saved to MongoDB with ID: {result.inserted_id}")
            except Exception as e:
                log.warning(f"Failed to save prediction to MongoDB: {e}")
        
        return jsonify(response_data), 200
        
    except Exception as e:
        log.error(f"Prediction error: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/prompt', methods=['POST'])
def prompt():
    """Обработка промпта из веб-интерфейса с сохранением в MongoDB"""
    try:
        if classifier is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        text = data.get("prompt", "").strip()
        if not text:
            return jsonify({'error': 'Empty prompt provided'}), 400

        # Парсим числа из промпта - с детальной обработкой ошибок
        try:
            # Разбиваем по запятым и удаляем пустые строки
            parts = [x.strip() for x in text.split(',') if x.strip()]
            
            if len(parts) != 60:
                return jsonify({
                    'error': f'Expected 60 numbers, got {len(parts)}',
                    'received_count': len(parts),
                    'expected_count': 60
                }), 400
            
            # Парсим каждое число отдельно для лучшей обработки ошибок
            arr = []
            for i, part in enumerate(parts):
                try:
                    num = float(part)
                    # Проверяем валидность числа (не NaN, не Inf)
                    if not (-1 <= num <= 1):
                        log.warning(f"Number {i} out of expected range [0,1]: {num}")
                    arr.append(num)
                except ValueError:
                    return jsonify({
                        'error': f'Invalid number at position {i}: "{part}"',
                        'position': i,
                        'value': part
                    }), 400
            
            features = np.array(arr).reshape(1, -1)

            # Предсказание
            prediction = classifier.predict(features)[0]
            probabilities = classifier.predict_proba(features)[0]

            log.info(f"Web prompt prediction: {prediction} with confidence {max(probabilities):.2%}")
            
            response_data = {
                'prediction': prediction,
                'probabilities': probabilities.tolist(),
                'confidence': {
                    'M': float(probabilities[0]),
                    'R': float(probabilities[1])
                },
                'interpretation': 'M = Mine (мина), R = Rock (скала)',
                'model_accuracy': '83.33%'
            }
            
            # Сохранение в MongoDB (если доступна БД)
            if mongodb_collection is not None:
                try:
                    db_record = {
                        'type': 'web_prompt',
                        'input_text': text,
                        'input_features': arr,
                        'prediction': prediction,
                        'probabilities': probabilities.tolist(),
                        'confidence': {
                            'M': float(probabilities[0]),
                            'R': float(probabilities[1])
                        },
                        'timestamp': datetime.utcnow()
                    }
                    result = mongodb_collection.insert_one(db_record)
                    response_data['db_id'] = str(result.inserted_id)
                    log.info(f"Web prompt prediction saved to MongoDB with ID: {result.inserted_id}")
                except Exception as e:
                    log.warning(f"Failed to save prompt prediction to MongoDB: {e}")
            
            return jsonify(response_data), 200
            
        except ValueError as e:
            log.error(f"Failed to parse numbers from prompt: {e}")
            return jsonify({
                'error': f'Failed to parse numbers: {str(e)}',
                'hint': 'Please provide 60 comma-separated decimal numbers'
            }), 400
        except Exception as e:
            log.error(f"Prompt processing error: {traceback.format_exc()}")
            return jsonify({'error': f'Error processing prompt: {str(e)}'}), 500
            
    except Exception as e:
        log.error(f"Prompt endpoint error: {traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500


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
        
        response_data = {
            'prediction': prediction,
            'confidence': {
                'M': float(probability[0]),
                'R': float(probability[1])
            },
            'interpretation': 'M = Mine (мина), R = Rock (скала)'
        }
        
        # Сохранение в MongoDB (если доступна БД)
        if mongodb_collection is not None:
            try:
                db_record = {
                    'type': 'predict_sonar',
                    'input_sonar_data': sonar_data,
                    'input_features': features.tolist(),
                    'prediction': prediction,
                    'confidence': {
                        'M': float(probability[0]),
                        'R': float(probability[1])
                    },
                    'timestamp': datetime.utcnow()
                }
                result = mongodb_collection.insert_one(db_record)
                response_data['db_id'] = str(result.inserted_id)
                log.info(f"Sonar prediction saved to MongoDB with ID: {result.inserted_id}")
            except Exception as e:
                log.warning(f"Failed to save prediction to MongoDB: {e}")
        
        return jsonify(response_data), 200
        
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
        
        response_data = {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'classes': classifier.classes_.tolist(),
            'total_samples': len(predictions)
        }
        
        # Сохранение в MongoDB (если доступна БД)
        if mongodb_collection is not None:
            try:
                db_record = {
                    'type': 'batch_predict',
                    'input_features': features.tolist(),
                    'predictions': predictions.tolist(),
                    'probabilities': probabilities.tolist(),
                    'total_samples': len(predictions),
                    'timestamp': datetime.utcnow()
                }
                result = mongodb_collection.insert_one(db_record)
                response_data['db_id'] = str(result.inserted_id)
                log.info(f"Batch prediction saved to MongoDB with ID: {result.inserted_id}")
            except Exception as e:
                log.warning(f"Failed to save batch prediction to MongoDB: {e}")
        
        return jsonify(response_data), 200
        
    except Exception as e:
        log.error(f"Batch prediction error: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/v1/save-prediction', methods=['POST'])
def save_prediction():
    """Сохраняет предсказание в MongoDB"""
    try:
        if mongodb_collection is None:
            return jsonify({'error': 'MongoDB connection not available'}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Добавляем временную метку
        data['timestamp'] = datetime.utcnow()
        
        # Вставляем документ в MongoDB
        result = mongodb_collection.insert_one(data)
        
        log.info(f"Prediction saved to MongoDB: {result.inserted_id}")
        return jsonify({
            'message': 'Prediction saved successfully',
            'id': str(result.inserted_id)
        }), 201
    except Exception as e:
        log.error(f"Error saving prediction: {traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/v1/get-predictions', methods=['GET'])
def get_predictions():
    """Получает все предсказания из MongoDB"""
    try:
        if mongodb_collection is None:
            return jsonify({'error': 'MongoDB connection not available'}), 503
        
        limit = request.args.get('limit', 10, type=int)
        skip = request.args.get('skip', 0, type=int)
        
        # Получаем предсказания из MongoDB с _id
        predictions = list(mongodb_collection.find({}).limit(limit).skip(skip))
        
        # Конвертируем ObjectId в строки
        for pred in predictions:
            if '_id' in pred:
                pred['_id'] = str(pred['_id'])
        
        log.info(f"Retrieved {len(predictions)} predictions from MongoDB")
        return jsonify({'predictions': predictions, 'count': len(predictions)}), 200
    except Exception as e:
        log.error(f"Error retrieving predictions: {traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/v1/get-predictions/<prediction_id>', methods=['GET'])
def get_prediction_by_id(prediction_id):
    """Получает предсказание по ID из MongoDB"""
    try:
        if mongodb_collection is None:
            return jsonify({'error': 'MongoDB connection not available'}), 503
        
        from bson.objectid import ObjectId
        
        prediction = mongodb_collection.find_one({'_id': ObjectId(prediction_id)})
        
        if not prediction:
            return jsonify({'error': 'Prediction not found'}), 404
        
        prediction['_id'] = str(prediction['_id'])
        
        log.info(f"Retrieved prediction: {prediction_id}")
        return jsonify(prediction), 200
    except Exception as e:
        log.error(f"Error retrieving prediction: {traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/v1/delete-prediction/<prediction_id>', methods=['DELETE'])
def delete_prediction(prediction_id):
    """Удаляет предсказание из MongoDB"""
    try:
        if mongodb_collection is None:
            return jsonify({'error': 'MongoDB connection not available'}), 503
        
        from bson.objectid import ObjectId
        
        result = mongodb_collection.delete_one({'_id': ObjectId(prediction_id)})
        
        if result.deleted_count == 0:
            return jsonify({'error': 'Prediction not found'}), 404
        
        log.info(f"Deleted prediction: {prediction_id}")
        return jsonify({'message': 'Prediction deleted successfully'}), 200
    except Exception as e:
        log.error(f"Error deleting prediction: {traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500


# Debugging: Print all available routes
print(app.url_map)

if __name__ == '__main__':
    if not load_model():
        log.warning("Starting without model - predictions will fail until model is loaded")
    
    connect_mongodb()
    
    log.info("Starting Flask API on 0.0.0.0:55566")
    app.run(host='0.0.0.0', port=55566, debug=False)
