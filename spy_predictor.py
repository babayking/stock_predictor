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
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('spy_predictor.log'),
        logging.StreamHandler()
    ]
)

class FinancialDataFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"
    
    def get_spy_data(self, from_date, to_date):
        """Fetch SPY historical data from Finnhub."""
        endpoint = f"{self.base_url}/stock/candle"
        
        params = {
            'symbol': 'SPY',
            'resolution': 'D',
            'from': int(from_date.timestamp()),
            'to': int(to_date.timestamp()),
            'token': self.api_key
        }
        
        try:
            response = requests.get(endpoint, params=params)
            if response.status_code == 200:
                data = response.json()
                if data.get('s') == 'ok':
                    df = pd.DataFrame({
                        'timestamp': data['t'],
                        'open': data['o'],
                        'high': data['h'],
                        'low': data['l'],
                        'close': data['c'],
                        'volume': data['v']
                    })
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    df.set_index('timestamp', inplace=True)
                    return df
            logging.error(f"Failed to fetch data: {response.text}")
            return None
        except Exception as e:
            logging.error(f"Error fetching data: {str(e)}")
            return None

class FinancialFeatureExtractor:
    @staticmethod
    def calculate_features(df):
        """Calculate technical indicators and features."""
        # Price changes
        df['returns'] = df['close'].pct_change()
        df['price_change'] = df['close'] - df['open']
        
        # Volatility (20-day)
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # RSI (14-day)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Clean up NaN values
        df.dropna(inplace=True)
        return df

class SPYPredictor:
    def __init__(
        self,
        feature_columns,
        num_actions=100,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995
    ):
        self.feature_columns = feature_columns
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        
        # State size is the number of features
        self.state_size = len(feature_columns)
        
        # Q-Network
        self.q_network = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions)
        )
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        
        # Initialize history
        self.history = {
            'actual_values': [],
            'predicted_values': [],
            'errors': [],
            'rewards': [],
            'epsilon': []
        }
        
        # Initialize scalers
        self.scalers = {}
    
    def fit_scalers(self, df):
        """Fit scalers to the feature data."""
        for column in self.feature_columns:
            min_val = df[column].min()
            max_val = df[column].max()
            self.scalers[column] = (min_val, max_val)
    
    def _normalize_features(self, features):
        """Normalize features using fitted scalers."""
        normalized = []
        for i, column in enumerate(self.feature_columns):
            min_val, max_val = self.scalers[column]
            value = (features[i] - min_val) / (max_val - min_val) if max_val > min_val else 0
            normalized.append(value)
        return normalized
    
    def _get_state(self, features):
        """Create state from features."""
        normalized_features = self._normalize_features(features)
        return torch.FloatTensor(normalized_features).unsqueeze(0)
    
    def _get_reward(self, prediction, actual):
        """Calculate reward based on prediction accuracy."""
        pct_error = abs((prediction - actual) / actual)
        return -pct_error
    
    def predict(self, features):
        """Choose action (prediction) using epsilon-greedy policy."""
        state = self._get_state(features)
        
        if np.random.random() < self.epsilon:
            action_idx = np.random.randint(0, self.num_actions)
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                action_idx = q_values.argmax().item()
        
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
        
        min_price = min(self.history['actual_values']) if self.history['actual_values'] else 0
        max_price = max(self.history['actual_values']) if self.history['actual_values'] else 100
        price_range = np.linspace(min_price, max_price, self.num_actions)
        
        return price_range[action_idx]
    
    def update(self, features, actual_value):
        """Update Q-network using the observed reward."""
        self.history['actual_values'].append(actual_value)
        
        prediction = self.predict(features)
        self.history['predicted_values'].append(prediction)
        
        reward = self._get_reward(prediction, actual_value)
        self.history['rewards'].append(reward)
        self.history['epsilon'].append(self.epsilon)
        
        error = abs(prediction - actual_value)
        self.history['errors'].append(error)
        
        self._train(features, reward)
        
        return error
    
    def _train(self, features, reward):
        """Train Q-network using current transition."""
        state = self._get_state(features)
        
        with torch.no_grad():
            next_q_values = self.q_network(state)
            next_q_value = next_q_values.max()
            target_q_value = reward + self.gamma * next_q_value
        
        q_values = self.q_network(state)
        current_q_value = q_values[0, q_values[0].argmax()]
        
        loss = (current_q_value - target_q_value) ** 2
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class ModelState:
    def __init__(self, base_path='model_state'):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        self.model_path = self.base_path / 'model.pth'
        self.scalers_path = self.base_path / 'scalers.pkl'
        self.history_path = self.base_path / 'history.pkl'
        self.metadata_path = self.base_path / 'metadata.json'
    
    def save_state(self, predictor, metadata):
        """Save complete model state."""
        torch.save(predictor.q_network.state_dict(), self.model_path)
        
        with open(self.scalers_path, 'wb') as f:
            pickle.dump(predictor.scalers, f)
        
        with open(self.history_path, 'wb') as f:
            pickle.dump(predictor.history, f)
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        logging.info(f"Model state saved at {datetime.now()}")
    
    def load_state(self, predictor):
        """Load model state if it exists."""
        if not all(p.exists() for p in [self.model_path, self.scalers_path, 
                                      self.history_path, self.metadata_path]):
            logging.info("No previous state found")
            return None
        
        predictor.q_network.load_state_dict(torch.load(self.model_path))
        
        with open(self.scalers_path, 'rb') as f:
            predictor.scalers = pickle.load(f)
        
        with open(self.history_path, 'rb') as f:
            predictor.history = pickle.load(f)
        
        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logging.info(f"Model state loaded from {metadata['last_update']}")
        return metadata

class PerformanceTracker:
    def __init__(self, base_path='performance'):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        self.daily_metrics_path = self.base_path / 'daily_metrics.csv'
        self.initialize_metrics_file()
    
    def initialize_metrics_file(self):
        if not self.daily_metrics_path.exists():
            pd.DataFrame(columns=[
                'date', 'mae', 'rmse', 'correct_direction_pct',
                'avg_return', 'sharpe_ratio'
            ]).to_csv(self.daily_metrics_path, index=False)
    
    def calculate_daily_metrics(self, predictor, actual_prices):
        predictions = predictor.history['predicted_values'][-len(actual_prices):]
        
        errors = np.abs(np.array(predictions) - np.array(actual_prices))
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(errors**2))
        
        actual_direction = np.diff(actual_prices) > 0
        predicted_direction = np.diff(predictions) > 0
        direction_accuracy = np.mean(actual_direction == predicted_direction)
        
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
        
        pd.DataFrame([metrics]).to_csv(
            self.daily_metrics_path, mode='a', header=False, index=False
        )
        
        return metrics

def run_continuous_training(api_key, days_to_run=30):
    """Run continuous training loop."""
    model_state = ModelState()
    performance_tracker = PerformanceTracker()
    
    predictor = SPYPredictor(feature_columns=[
        'returns', 'volatility', 'rsi', 'macd',
        'macd_signal', 'price_change'
    ])
    
    metadata = model_state.load_state(predictor)
    last_update = (datetime.fromisoformat(metadata['last_update']) 
                  if metadata else datetime.now() - timedelta(days=1))
    
    fetcher = FinancialDataFetcher(api_key)
    
    # Get initial data and fit scalers
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Get last 30 days for initial scaling
    initial_data = fetcher.get_spy_data(start_date, end_date)
    
    if initial_data is not None:
        initial_data = FinancialFeatureExtractor.calculate_features(initial_data)
        predictor.fit_scalers(initial_data)
    
    while (datetime.now() - last_update).days <= days_to_run:
        try:
            end_date = datetime.now()
            start_date = last_update
            
            logging.info(f"Fetching data from {start_date} to {end_date}")
            df = fetcher.get_spy_data(start_date, end_date)
            
            if df is None or df.empty:
                logging.warning("No new data available")
                time.sleep(3600)
                continue
            
            df = FinancialFeatureExtractor.calculate_features(df)
            
            for i in range(len(df) - 1):
                features = df.iloc[i][predictor.feature_columns].values
                next_price = df.iloc[i+1]['close']
                error = predictor.update(features, next_price)
            
            daily_metrics = performance_tracker.calculate_daily_metrics(
                predictor, df['close'].values
            )
            
            metadata = {
                'last_update': end_date.isoformat(),
                'total_training_steps': len(predictor.history['errors']),
                'current_mae': daily_metrics['mae'],
                'current_direction_accuracy': daily_metrics['correct_direction_pct']
            }
            model_state.save_state(predictor, metadata)
            
            logging.info(f"Training metrics for {end_date.date()}:")
            logging.info(f"MAE: ${daily_metrics['mae']:.2f}")
            logging.info(f"Direction Accuracy: {daily_metrics['correct_direction_pct']:.2%}")
            
            time.sleep(3600)
            
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            time.sleep(3600)

if __name__ == "__main__":
    # Get API key from environment variable
    api_key = os.getenv('FINNHUB_API_KEY')
    if not api_key:
        raise ValueError("Please set FINNHUB_API_KEY environment variable")
    
    logging.info("Starting SPY predictor training (batch mode)...")
    try:
        # Create initial directories if they don't exist
        os.makedirs('model_state', exist_ok=True)
        os.makedirs('performance', exist_ok=True)
        
        # Setup objects
        model_state = ModelState()
        performance_tracker = PerformanceTracker()
        predictor = SPYPredictor(feature_columns=[
            'returns', 'volatility', 'rsi', 'macd',
            'macd_signal', 'price_change'
        ])
        
        # Load previous state if available
        metadata = model_state.load_state(predictor)
        last_update = (datetime.fromisoformat(metadata['last_update']) 
                    if metadata else datetime.now() - timedelta(days=30))
        
        # Get a single batch of data
        fetcher = FinancialDataFetcher(api_key)
        end_date = datetime.now()
        
        logging.info(f"Fetching data from {last_update} to {end_date}")
        df = fetcher.get_spy_data(last_update, end_date)
        
        if df is not None and not df.empty:
            # Process this batch
            df = FinancialFeatureExtractor.calculate_features(df)
            
            # If first run, fit scalers
            if not predictor.scalers:
                predictor.fit_scalers(df)
            
            # Run training on this batch
            for i in range(len(df) - 1):
                features = df.iloc[i][predictor.feature_columns].values
                next_price = df.iloc[i+1]['close']
                error = predictor.update(features, next_price)
            
            # Calculate metrics
            daily_metrics = performance_tracker.calculate_daily_metrics(
                predictor, df['close'].values
            )
            
            # Save state for next run
            metadata = {
                'last_update': end_date.isoformat(),
                'total_training_steps': len(predictor.history['errors']),
                'current_mae': daily_metrics['mae'],
                'current_direction_accuracy': daily_metrics['correct_direction_pct']
            }
            model_state.save_state(predictor, metadata)
            
            # Log results
            logging.info(f"Training metrics for {end_date.date()}:")
            logging.info(f"MAE: ${daily_metrics['mae']:.2f}")
            logging.info(f"Direction Accuracy: {daily_metrics['correct_direction_pct']:.2%}")
        else:
            logging.warning("No new data available")
            
    except Exception as e:
        logging.error(f"Error in batch execution: {str(e)}")
        raise e
