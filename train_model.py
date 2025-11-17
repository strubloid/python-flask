"""
Diabetes Prediction Model Training Pipeline

This module provides a comprehensive machine learning pipeline for training
a K-Nearest Neighbors classifier to predict diabetes risk based on the
Pima Indians Diabetes Database.

Data source: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class DiabetesModelTrainer:
    """
    A comprehensive trainer class for building and evaluating a diabetes prediction model.
    
    This class encapsulates the entire machine learning pipeline including:
    - Data loading and validation
    - Feature preprocessing
    - Model training with cross-validation
    - Model evaluation
    - Model persistence
    
    Attributes:
        data_path (str): Path to the CSV file containing the diabetes dataset
        test_size (float): Proportion of dataset to include in the test split
        random_state (int): Random state for reproducibility
        n_neighbors (int): Number of neighbors for KNN classifier
        model (KNeighborsClassifier): The trained KNN model
        scaler (StandardScaler): Fitted scaler for feature normalization
        X_train, X_test, y_train, y_test: Train/test splits
    """
    
    def __init__(
        self,
        data_path: str = 'diabetes.csv',
        test_size: float = 0.2,
        random_state: int = 42,
        n_neighbors: int = 5
    ):
        """
        Initialize the DiabetesModelTrainer with configuration parameters.
        
        Args:
            data_path: Path to the diabetes dataset CSV file
            test_size: Fraction of data to reserve for testing (0.0 to 1.0)
            random_state: Seed for random number generator (for reproducibility)
            n_neighbors: Number of neighbors to use in KNN algorithm
        """
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        
        # Initialize placeholders
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the diabetes dataset from CSV file.
        
        Returns:
            DataFrame containing the loaded data
            
        Raises:
            FileNotFoundError: If the data file doesn't exist
            ValueError: If the data is invalid or empty
        """
        print(f"Loading data from {self.data_path}...")
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return self.df
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")
    
    def validate_data(self) -> None:
        """
        Validate the loaded dataset for completeness and correctness.
        
        Raises:
            ValueError: If data validation fails
        """
        if self.df is None or self.df.empty:
            raise ValueError("Dataset is empty or not loaded")
        
        # Check for required columns
        expected_features = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
        ]
        
        missing_cols = set(expected_features) - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for null values
        null_counts = self.df.isnull().sum()
        if null_counts.any():
            print("Warning: Null values detected:")
            print(null_counts[null_counts > 0])
        
        print("Data validation passed ✓")
    
    def prepare_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split the dataset into features (X) and target variable (y).
        
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        print("Preparing features and target variable...")
        X = self.df.drop('Outcome', axis=1)
        y = self.df['Outcome']
        
        print(f"Features shape: {X.shape}")
        print(f"Target distribution:\n{y.value_counts()}")
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature matrix
            y: Target variable
        """
        print(f"\nSplitting data (test_size={self.test_size}, random_state={self.random_state})...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y  # Maintain class distribution in splits
        )
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Testing set: {self.X_test.shape[0]} samples")
    
    def train_model(self) -> KNeighborsClassifier:
        """
        Train the K-Nearest Neighbors classifier.
        
        Returns:
            Trained KNN model
        """
        print(f"\nTraining KNN model (n_neighbors={self.n_neighbors})...")
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.model.fit(self.X_train, self.y_train)
        print("Model training completed ✓")
        
        return self.model
    
    def evaluate_model(self) -> Dict[str, float]:
        """
        Evaluate the trained model using multiple metrics.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Training accuracy
        train_score = self.model.score(self.X_train, self.y_train)
        print(f"\nTraining Accuracy: {train_score:.4f}")
        
        # Testing accuracy
        test_score = self.model.score(self.X_test, self.y_test)
        print(f"Testing Accuracy: {test_score:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, self.X_train, self.y_train,
            cv=5, scoring='accuracy'
        )
        print(f"\nCross-Validation Scores: {cv_scores}")
        print(f"CV Mean Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Predictions on test set
        y_pred = self.model.predict(self.X_test)
        
        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        print(f"True Negatives: {cm[0][0]}, False Positives: {cm[0][1]}")
        print(f"False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}")
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    def save_model(self, model_path: str = 'example_weights_knn.pkl') -> None:
        """
        Save the trained model to disk using pickle.
        
        Args:
            model_path: Path where the model will be saved
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        print(f"\nSaving model to {model_path}...")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved successfully ✓")
    
    def process(self, save_path: str = 'example_weights_knn.pkl') -> Dict[str, float]:
        """
        Execute the complete machine learning pipeline.
        
        This method orchestrates the entire workflow:
        1. Load data
        2. Validate data
        3. Prepare features
        4. Split into train/test sets
        5. Train model
        6. Evaluate performance
        7. Save model
        
        Args:
            save_path: Path to save the trained model
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print("\n" + "="*60)
        print("DIABETES PREDICTION MODEL TRAINING PIPELINE")
        print("="*60 + "\n")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Validate data
        self.validate_data()
        
        # Step 3: Prepare features
        X, y = self.prepare_features()
        
        # Step 4: Split data
        self.split_data(X, y)
        
        # Step 5: Train model
        self.train_model()
        
        # Step 6: Evaluate model
        metrics = self.evaluate_model()
        
        # Step 7: Save model
        self.save_model(save_path)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        
        return metrics


def main():
    """
    Main entry point for training the diabetes prediction model.
    """
    # Initialize trainer with configuration
    trainer = DiabetesModelTrainer(
        data_path='diabetes.csv',
        test_size=0.2,
        random_state=42,
        n_neighbors=5
    )
    
    # Execute the complete training pipeline
    metrics = trainer.process(save_path='example_weights_knn.pkl')
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL METRICS SUMMARY")
    print("="*60)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
