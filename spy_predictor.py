import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import requests
from datetime import datetime, timedelta
import json
import os
import logging
import pickle
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('spy_predictor.log'),
        logging.StreamHandler()
    ]
)

class ModelState:
    """Class to handle model state persistence."""
    def __init__(self, base_path='model_state'):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        self.model_path = self.base_path / 'model.pth'
        self.scalers_path = self.base_path / 'scalers.pkl'
        self.history_path = self.base_path / 'history.pkl'
        self.metadata_path = self.base_path / 'metadata.json'
    
    def save_state(self, predictor, metadata):
        """Save complete model state."""
        # Save PyTorch model
        torch.save(predictor.q_network.state_dict(), self.model_path)
        
        # Save scalers
        with open(self.scalers_path, 'wb') as f:
            pickle.dump(predictor.scalers, f)
        
        # Save history
        with open(self.history_path, 'wb') as f:
            pickle.dump(predictor.history, f)
        
        # Save metadata
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        logging.info(f"Model state saved at {datetime.now()}")
    
    def load_state(self, predictor):
        """Load model state if it exists."""
        if not all(p.exists() for p in [self.model_path, self.scalers_path, 
                                      self.history_path, self.metadata_path]):
            logging.info("No previous state found")
            return None
        
        # Load PyTorch model
        predictor.q_network.load_state_dict(
            torch.load(self.model_path)
        )
        
        # Load scalers
        with open(self.scalers_path, 'rb') as f:
            predictor.scalers = pickle.load(f)
        
        # Load history
        with open(self.history_path, 'rb') as f:
            predictor.history = pickle.load(f)
        
        # Load metadata
        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logging.info(f"Model state loaded from {metadata['last_update']}")
        return metadata

class PerformanceTracker:
    """Track and analyze model performance over time."""
    def __init__(self, base_path='performance'):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        self.daily_metrics_path = self.base_path / 'daily_metrics.csv'
        self.initialize_metrics_file()
    
    def initialize_metrics_file(self):
        """Create metrics file if it doesn't exist."""
        if not self.daily_metrics_path.exists():
            pd.DataFrame(columns=[
                'date', 'mae', 'rmse', 'correct_direction_pct',
                'avg_return', 'sharpe_ratio'
            ]).to_csv(self.daily_metrics_path, index=False)
    
    def calculate_daily_metrics(self, predictor, actual_prices):
        """Calculate performance metrics for the day."""
        predictions = predictor.history['predicted_values'][-len(actual_prices):]
        
        # Basic error metrics
        errors = np.abs(np.array(predictions) - np.array(actual_prices))
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(errors**2))
        
        # Direction prediction accuracy
        actual_direction = np.diff(actual_prices) > 0
        predicted_direction = np.diff(predictions) > 0
        direction_accuracy = np.mean(actual_direction == predicted_direction)
        
        # Return metrics
        returns = np.diff(actual_prices) / actual_prices[:-1]
        avg_return = np.mean(returns)
        sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 else 0
        
        metrics = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'mae': mae,
            'rmse': rmse,
            'correct_direction_pct': direction_accuracy,
            'avg_return': avg_return,
            'sharpe_ratio': sharpe_ratio
        }
        
        # Append to CSV
        pd.DataFrame([metrics]).to_csv(
            self.daily_metrics_path, mode='a', header=False, index=False
        )
        
        return metrics
    
    def plot_performance_evolution(self):
        """Plot how model performance has evolved over time."""
        metrics_df = pd.read_csv(self.daily_metrics_path)
        metrics_df['date'] = pd.to_datetime(metrics_df['date'])
        
        plt.figure(figsize=(15, 10))
        
        # Plot MAE over time
        plt.subplot(2, 2, 1)
        plt.plot(metrics_df['date'], metrics_df['mae'])
        plt.title('Mean Absolute Error Over Time')
        plt.xticks(rotation=45)
        plt.grid(True)
        
        # Plot direction prediction accuracy
        plt.subplot(2, 2, 2)
        plt.plot(metrics_df['date'], metrics_df['correct_direction_pct'])
        plt.title('Direction Prediction Accuracy Over Time')
        plt.xticks(rotation=45)
        plt.grid(True)
        
        # Plot Sharpe ratio
        plt.subplot(2, 2, 3)
        plt.plot(metrics_df['date'], metrics_df['sharpe_ratio'])
        plt.title('Sharpe Ratio Over Time')
        plt.xticks(rotation=45)
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.base_path / 'performance_evolution.png')
        plt.close()

def create_github_workflow():
    """Create GitHub Actions workflow file."""
    workflow_dir = Path('.github/workflows')
    workflow_dir.mkdir(parents=True, exist_ok=True)
    
    workflow_content = """
name: SPY Predictor Training

on:
  schedule:
    - cron: '0 0 * * *'  # Run daily at midnight UTC
  workflow_dispatch:  # Allow manual triggers

jobs:
  train:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install torch pandas numpy requests matplotlib
    
    - name: Run predictor
      env:
        FINNHUB_API_KEY: ${{ secrets.FINNHUB_API_KEY }}
      run: python spy_predictor.py
    
    - name: Commit changes
      run: |
        git config --local user.email "action@github.com"
        git config --local user.name "GitHub Action"
        git add model_state/* performance/*
        git commit -m "Update model state and performance metrics" || echo "No changes to commit"
        git push
    """
    
    with open(workflow_dir / 'spy_predictor.yml', 'w') as f:
        f.write(workflow_content.strip())

def run_continuous_training(api_key, days_to_run=30):
    """Run continuous training loop."""
    model_state = ModelState()
    performance_tracker = PerformanceTracker()
    
    # Initialize or load predictor
    predictor = SPYPredictor(feature_columns=[
        'returns', 'volatility', 'rsi', 'macd',
        'macd_signal', 'price_change'
    ])
    
    metadata = model_state.load_state(predictor)
    last_update = (datetime.fromisoformat(metadata['last_update']) 
                  if metadata else datetime.now() - timedelta(days=1))
    
    # Create data fetcher
    fetcher = FinancialDataFetcher(api_key)
    
    while (datetime.now() - last_update).days <= days_to_run:
        try:
            # Get latest data
            end_date = datetime.now()
            start_date = last_update
            
            logging.info(f"Fetching data from {start_date} to {end_date}")
            df = fetcher.get_spy_data(start_date, end_date)
            
            if df is None or df.empty:
                logging.warning("No new data available")
                time.sleep(3600)  # Wait an hour before trying again
                continue
            
            # Calculate features
            df = FinancialFeatureExtractor.calculate_features(df)
            
            # Update predictor
            for i in range(len(df) - 1):
                features = df.iloc[i][predictor.feature_columns].values
                next_price = df.iloc[i+1]['close']
                error = predictor.update(features, next_price)
            
            # Calculate and save daily metrics
            daily_metrics = performance_tracker.calculate_daily_metrics(
                predictor, df['close'].values
            )
            
            # Update plots
            performance_tracker.plot_performance_evolution()
            
            # Save model state
            metadata = {
                'last_update': end_date.isoformat(),
                'total_training_steps': len(predictor.history['errors']),
                'current_mae': daily_metrics['mae'],
                'current_direction_accuracy': daily_metrics['correct_direction_pct']
            }
            model_state.save_state(predictor, metadata)
            
            # Log progress
            logging.info(f"Training metrics for {end_date.date()}:")
            logging.info(f"MAE: ${daily_metrics['mae']:.2f}")
            logging.info(f"Direction Accuracy: {daily_metrics['correct_direction_pct']:.2%}")
            
            # Wait until next day
            time.sleep(3600)  # Check every hour for new data
            
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            time.sleep(3600)  # Wait an hour before retrying

if __name__ == "__main__":
    # Setup GitHub workflow
    create_github_workflow()
    
    # Get API key from environment variable
    api_key = os.getenv('FINNHUB_API_KEY')
    if not api_key:
        raise ValueError("Please set FINNHUB_API_KEY environment variable")
    
    # Run continuous training
    run_continuous_training(api_key)
