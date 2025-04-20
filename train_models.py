from ml_models import TeamIdentifier, SignalReliability, MarketMovementPredictor, BetSizer
from pathlib import Path
import argparse
import json
from datetime import datetime
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Dict, Optional
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW

def load_training_data(matches_file: str, historical_file: Optional[str] = None) -> tuple:
    """
    Load training data from JSON files
    
    Args:
        matches_file (str): Path to matches JSON file
        historical_file (Optional[str]): Path to historical performance JSON file
        
    Returns:
        tuple: (matches_data, historical_data)
    """
    try:
        with open(matches_file, 'r') as f:
            matches = json.load(f)
            
        historical_data = {}
        if historical_file:
            with open(historical_file, 'r') as f:
                historical_data = json.load(f)
                
        return matches, historical_data
        
    except Exception as e:
        print(f"Error loading training data: {str(e)}")
        raise

def train_team_identifier(matches: List[Dict], output_dir: str) -> None:
    """
    Train the team identifier model
    
    Args:
        matches (List[Dict]): List of match data
        output_dir (str): Directory to save the trained model
    """
    try:
        # Extract training data
        texts = []
        teams = []
        for match in matches:
            # Get text from title and body
            text = f"{match.get('title', '')} {match.get('text', '')} {match.get('comment_text', '')}"
            if text.strip():
                # Get team from subreddit
                team = match.get('subreddit', '').lower()
                if team in ['reddevils', 'gunners', 'coys', 'mcfc', 'chelseafc', 'liverpoolfc']:
                    texts.append(text)
                    teams.append(1 if team == 'home' else 0)
        
        if not texts:
            print("No training data available for team identifier")
            return
            
        # Initialize and train model
        model = TeamIdentifier()
        model.fit(texts, teams)
        
        # Save model
        output_path = Path(output_dir) / 'team_identifier'
        output_path.mkdir(parents=True, exist_ok=True)
        model.model.save_pretrained(output_path)
        model.tokenizer.save_pretrained(output_path)
        
        print(f"Team identifier model saved to {output_path}")
        
    except Exception as e:
        print(f"Error training team identifier: {str(e)}")
        raise

def train_signal_reliability(matches: List[Dict], historical_data: Dict, output_dir: str) -> None:
    """
    Train the signal reliability model
    
    Args:
        matches (List[Dict]): List of match data
        historical_data (Dict): Historical performance data
        output_dir (str): Directory to save the trained model
    """
    try:
        # Extract training data
        features = []
        labels = []
        for match in matches:
            # Extract features from post metadata
            feature = {
                'source': match.get('subreddit', ''),
                'content': f"{match.get('title', '')} {match.get('text', '')}",
                'sentiment': match.get('sentiment', 'neutral'),
                'score': match.get('score', 0),
                'num_comments': match.get('num_comments', 0)
            }
            
            # Use score and comments as proxy for reliability
            label = 1.0 if feature['score'] > 10 and feature['num_comments'] > 5 else 0.0
            
            features.append(feature)
            labels.append(label)
        
        if not features:
            print("No training data available for signal reliability")
            return
            
        # Initialize and train model
        model = SignalReliability()
        model.fit(features, labels)
        
        # Save model
        output_path = Path(output_dir) / 'signal_reliability'
        model.save(output_path)
        
        print(f"Signal reliability model saved to {output_path}")
        
    except Exception as e:
        print(f"Error training signal reliability: {str(e)}")
        raise

def train_market_movement(matches: List[Dict], output_dir: str) -> None:
    """
    Train the market movement predictor model
    
    Args:
        matches (List[Dict]): List of match data
        output_dir (str): Directory to save the trained model
    """
    try:
        # Extract training data
        features = []
        labels = []
        for match in matches:
            # For now, we'll use sentiment and engagement metrics as features
            if match.get('sentiment') and match.get('score') is not None:
                sentiment_value = {
                    'positive': 1.0,
                    'neutral': 0.5,
                    'negative': 0.0
                }.get(match['sentiment'], 0.5)
                
                features.append({
                    'sentiment': sentiment_value,
                    'score': match['score'],
                    'num_comments': match.get('num_comments', 0)
                })
                # For now, use a simple binary label based on engagement
                labels.append(1 if match.get('score', 0) > 20 else 0)
        
        if not features:
            print("No training data available for market movement")
            return
            
        # Initialize and train model
        model = MarketMovementPredictor()
        model.fit(features, labels)
        
        # Save model
        output_path = Path(output_dir) / 'market_movement'
        model.save(output_path)
        
        print(f"Market movement model saved to {output_path}")
        
    except Exception as e:
        print(f"Error training market movement: {str(e)}")
        raise

def train_bet_sizer(matches: List[Dict], historical_data: Dict, output_dir: str) -> None:
    """
    Train the bet sizer model
    
    Args:
        matches (List[Dict]): List of match data
        historical_data (Dict): Historical performance data
        output_dir (str): Directory to save the trained model
    """
    try:
        # Extract training data
        features = []
        labels = []
        for match in matches:
            if match.get('sentiment') and match.get('score') is not None:
                # Create feature vector from available data
                feature = {
                    'sentiment': match['sentiment'],
                    'score': match['score'],
                    'num_comments': match.get('num_comments', 0),
                    'subreddit': match.get('subreddit', ''),
                    'has_url': bool(match.get('url'))
                }
                
                # For now, use a simple stake sizing based on confidence metrics
                confidence = min(1.0, (match.get('score', 0) / 100) + 
                               (0.5 if match['sentiment'] == 'positive' else 0.0))
                
                features.append(feature)
                labels.append(confidence)
        
        if not features:
            print("No training data available for bet sizer")
            return
            
        # Initialize and train model
        model = BetSizer()
        model.fit(features, labels)
        
        # Save model
        output_path = Path(output_dir) / 'bet_sizer'
        model.save(output_path)
        
        print(f"Bet sizer model saved to {output_path}")
        
    except Exception as e:
        print(f"Error training bet sizer: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Train WhisperBet models')
    parser.add_argument('--matches', type=str, required=True, help='Path to matches JSON file')
    parser.add_argument('--historical', type=str, help='Path to historical performance JSON file')
    parser.add_argument('--output', type=str, default='models', help='Output directory for trained models')
    
    args = parser.parse_args()
    
    try:
        # Load training data
        matches, historical_data = load_training_data(args.matches, args.historical)
        
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Train models
        print("Training team identifier...")
        train_team_identifier(matches, args.output)
        
        print("\nTraining signal reliability...")
        train_signal_reliability(matches, historical_data, args.output)
        
        print("\nTraining market movement predictor...")
        train_market_movement(matches, args.output)
        
        print("\nTraining bet sizer...")
        train_bet_sizer(matches, historical_data, args.output)
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == '__main__':
    main() 