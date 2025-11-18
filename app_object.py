"""
Flask Application with Object-Oriented Design and Factory Pattern
This module implements a diabetes prediction web service using OOP principles.
"""

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pandas as pd
import pickle
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional


class ModelLoader(ABC):
    """Abstract base class for model loading strategies."""
    
    @abstractmethod
    def load_model(self, model_path: str):
        """Load and return the model."""
        pass


class PickleModelLoader(ModelLoader):
    """Concrete implementation for loading pickle models."""
    
    def load_model(self, model_path: str):
        """Load a pickle model from the specified path."""
        try:
            with open(model_path, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")


class FormValidator:
    """Handles form data validation."""
    
    REQUIRED_FIELDS = [
        'pregnancies', 'glucose', 'blood_presure', 'skin_thickness',
        'insulin_level', 'bmi', 'diabetes_pedigree', 'age'
    ]
    
    @classmethod
    def validate(cls, form_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate that all required fields are present and not empty.
        
        Args:
            form_data: Dictionary containing form data
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        for field in cls.REQUIRED_FIELDS:
            if field not in form_data or not str(form_data[field]).strip():
                return False, f"Field '{field}' is required and cannot be empty"
        
        return True, None
    
    @classmethod
    def get_required_fields(cls):
        """Return list of required fields."""
        return cls.REQUIRED_FIELDS.copy()


class DataPreprocessor:
    """Handles data preprocessing for model input."""
    
    @staticmethod
    def prepare_dataframe(form_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert form data to pandas DataFrame for model prediction.
        
        Args:
            form_data: Dictionary containing validated form data
            
        Returns:
            DataFrame ready for model prediction
        """
        values = [
            form_data['pregnancies'],
            form_data['glucose'],
            form_data['blood_presure'],
            form_data['skin_thickness'],
            form_data['insulin_level'],
            form_data['bmi'],
            form_data['diabetes_pedigree'],
            form_data['age']
        ]
        
        return pd.DataFrame([values])


class PredictionService:
    """Handles prediction logic and result formatting."""
    
    def __init__(self, model):
        """
        Initialize prediction service with a trained model.
        
        Args:
            model: Trained machine learning model
        """
        self.model = model
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Make prediction on input data.
        
        Args:
            data: Preprocessed DataFrame
            
        Returns:
            Dictionary containing prediction results
        """
        # Get probability predictions
        probabilities = self.model.predict_proba(data)
        
        # Get probability of positive class (diabetes)
        positive_probability = probabilities[0][1]
        
        return self._format_prediction_result(positive_probability)
    
    @staticmethod
    def _format_prediction_result(probability: float) -> Dict[str, Any]:
        """
        Format prediction result for response.
        
        Args:
            probability: Probability of positive class
            
        Returns:
            Formatted prediction dictionary
        """
        probability_percentage = probability * 100
        formatted_percentage = '{0:.{1}f}'.format(probability_percentage, 2)
        
        prediction_class = 'positive' if probability > 0.5 else 'negative'
        risk_level = 'a high' if probability > 0.5 else 'a low'
        
        return {
            'prediction': prediction_class,
            'probability': float(formatted_percentage),
            'probability_formatted': f'{formatted_percentage}%',
            'message': f'You have {risk_level} chance of having diabetes: {formatted_percentage}%'
        }


class RouteHandler:
    """Handles Flask route logic."""
    
    def __init__(self, prediction_service: PredictionService):
        """
        Initialize route handler with prediction service.
        
        Args:
            prediction_service: Instance of PredictionService
        """
        self.prediction_service = prediction_service
        self.validator = FormValidator()
        self.preprocessor = DataPreprocessor()
    
    def handle_index(self):
        """Handle the index route."""
        return render_template('index.html')
    
    def handle_predict(self):
        """Handle the prediction route."""
        # Validate form data
        is_valid, error_message = self.validator.validate(request.form)
        
        if not is_valid:
            return render_template('result.html', pred=f'Error: {error_message}'), 400
        
        try:
            # Preprocess data
            input_data = self.preprocessor.prepare_dataframe(request.form)
            
            # Make prediction
            prediction_result = self.prediction_service.predict(input_data)
            
            return jsonify(prediction_result)
            
        except Exception as e:
            return jsonify({
                'error': 'Prediction failed',
                'message': str(e)
            }), 500


class CORSConfig:
    """Configuration for CORS settings."""
    
    @staticmethod
    def get_config() -> Dict[str, Any]:
        """Return CORS configuration."""
        return {
            r"/*": {
                "origins": [
                    "https://ds-frontend-8b124c0b64a5.herokuapp.com",
                    "http://localhost:*",
                    "http://127.0.0.1:*"
                ],
                "methods": ["GET", "POST", "OPTIONS"],
                "allow_headers": ["Content-Type"]
            }
        }


class FlaskAppFactory:
    """Factory class for creating Flask application instances."""
    
    def __init__(self, model_path: str = 'example_weights_knn.pkl'):
        """
        Initialize factory with model path.
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path
        self.model_loader = PickleModelLoader()
    
    def create_app(self) -> Flask:
        """
        Create and configure Flask application.
        
        Returns:
            Configured Flask application instance
        """
        app = Flask(__name__)
        
        # Configure CORS
        self._configure_cors(app)
        
        # Load model
        model = self._load_model()
        
        # Create services
        prediction_service = PredictionService(model)
        route_handler = RouteHandler(prediction_service)
        
        # Register routes
        self._register_routes(app, route_handler)
        
        return app
    
    def _configure_cors(self, app: Flask) -> None:
        CORS(app, resources=CORSConfig.get_config())
    
    def _load_model(self):
        return self.model_loader.load_model(self.model_path)
    
    @staticmethod
    def _register_routes(app: Flask, route_handler: RouteHandler) -> None:
        app.add_url_rule(
            '/',
            'index',
            route_handler.handle_index,
            methods=['GET']
        )
        
        app.add_url_rule(
            '/predict',
            'predict',
            route_handler.handle_predict,
            methods=['POST', 'GET']
        )


# Application entry point
def main():
    """Main entry point for the application."""
    # Create application using factory
    factory = FlaskAppFactory(model_path='example_weights_knn.pkl')
    app = factory.create_app()
    
    # Run application
    app.run(host='0.0.0.0', port=5000, debug=True)


if __name__ == '__main__':
    main()
