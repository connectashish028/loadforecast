"""
Model Evaluation and Prediction Script
This script loads the trained model and performs evaluation and prediction tasks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
import os
import csv

def load_model_and_data():
    """Load the trained model and test data."""
    try:
        # Load the model (using the best performing model)
        model = load_model('model_checkpoints/best_Conv1D.h5', compile=False)
        
        # Load test data
        X_test = np.load('X_test.npy')
        y_test = np.load('y_test.npy')
        
        return model, X_test, y_test
    except Exception as e:
        print(f"Error loading model or data: {str(e)}")
        raise

def evaluate_and_save(model_name, model_path, X_test, y_test, metrics_list):
    print(f"\nEvaluating {model_name}...")
    model = load_model(model_path, compile=False)
    y_pred = model.predict(X_test)
    y_pred_flat = y_pred.flatten()
    y_test_flat = y_test.flatten()
    mse = mean_squared_error(y_test_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_flat, y_pred_flat)
    r2 = r2_score(y_test_flat, y_pred_flat)
    print(f"{model_name} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    metrics_list.append({
        'Model': model_name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    })
    plot_predictions(y_test, y_pred, save_path=f'output/predictions_{model_name}.png')
    plot_residuals(y_test, y_pred, save_path=f'output/residuals_{model_name}.png')

def plot_predictions(y_test, y_pred, save_path='output/predictions.png'):
    """Create visualization of actual vs predicted values."""
    try:
        # Flatten arrays for plotting
        y_test_flat = y_test.flatten()
        y_pred_flat = y_pred.flatten()
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_flat, label='Actual', color='blue', alpha=0.7)
        plt.plot(y_pred_flat, label='Predicted', color='red', alpha=0.7)
        plt.title('Actual vs Predicted Energy Load')
        plt.xlabel('Time')
        plt.ylabel('Load (MW)')
        plt.legend()
        plt.grid(True)
        
        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)
        
        # Save the plot
        plt.savefig(save_path)
        plt.close()
        print(f"\nPrediction plot saved to {save_path}")
    except Exception as e:
        print(f"Error creating prediction plot: {str(e)}")
        raise

def plot_residuals(y_test, y_pred, save_path='output/residuals.png'):
    """Create visualization of prediction residuals."""
    try:
        # Flatten arrays for residual calculation
        y_test_flat = y_test.flatten()
        y_pred_flat = y_pred.flatten()
        residuals = y_test_flat - y_pred_flat
        
        plt.figure(figsize=(12, 6))
        plt.scatter(y_pred_flat, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residual Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.grid(True)
        
        # Save the plot
        plt.savefig(save_path)
        plt.close()
        print(f"\nResidual plot saved to {save_path}")
    except Exception as e:
        print(f"Error creating residual plot: {str(e)}")
        raise

def main():
    print("Starting model comparison...")
    model_files = {
        'Conv1D': 'model_checkpoints/best_Conv1D.h5',
        'LSTM': 'model_checkpoints/best_LSTM.h5',
        'GRU': 'model_checkpoints/best_GRU.h5',
    }
    model, X_test, y_test = load_model_and_data()  # Just to load X_test, y_test
    metrics_list = []
    for model_name, model_path in model_files.items():
        evaluate_and_save(model_name, model_path, X_test, y_test, metrics_list)
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv('output/model_comparison_metrics.csv', index=False)
    print("\nModel comparison complete! Metrics saved to output/model_comparison_metrics.csv")
    print("Check the output directory for all plots.")

if __name__ == "__main__":
    main() 