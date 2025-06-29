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

def load_training_data(matches_file, historical_file=None):
    # load data from json files for training
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

def train_team_identifier(matches, output_dir):
    # train model to identify which team a post is about
    try:
        texts = []
        teams = []
        for match in matches:
            # combine title and body text
            text = f"{match.get('title', '')} {match.get('text', '')} {match.get('comment_text', '')}"
            if text.strip():
                # get team from subreddit - pretty basic but works
                team = match.get('subreddit', '').lower()
                if team in ['reddevils', 'gunners', 'coys', 'mcfc', 'chelseafc', 'liverpoolfc']:
                    texts.append(text)
                    teams.append(1 if team == 'home' else 0)  # simplified binary classification
        
        if not texts:
            print("No training data available for team identifier")
            return
            
        # train the model
        model = TeamIdentifier()
        model.fit(texts, teams)
        
        # save it
        output_path = Path(output_dir) / 'team_identifier'
        output_path.mkdir(parents=True, exist_ok=True)
        model.model.save_pretrained(output_path)
        model.tokenizer.save_pretrained(output_path)
        
        print(f"Team identifier model saved to {output_path}")
        
    except Exception as e:
        print(f"Error training team identifier: {str(e)}")

def train_signal_reliability(matches, historical_data, output_dir):
    # train model to score signal reliability
    
    features = []
    labels = []
    for match in matches:
        # extract basic features from post
        feature = {
            'source': match.get('subreddit', ''),
            'content': f"{match.get('title', '')} {match.get('text', '')}",
            'sentiment': match.get('sentiment', 'neutral'),
            'score': match.get('score', 0),
            'num_comments': match.get('num_comments', 0)
        }
        
        # use score and comments as proxy for reliability - not perfect but ok for now
        label = 1.0 if feature['score'] > 10 and feature['num_comments'] > 5 else 0.0
        
        features.append(feature)
        labels.append(label)
    
    if not features:
        print("No training data available for signal reliability")
        return
        
    # train model
    model = SignalReliability()
    model.fit(features, labels)
    
    # save model
    output_path = Path(output_dir) / 'signal_reliability'
    model.save(output_path)
    
    print(f"Signal reliability model saved to {output_path}")

def train_market_movement(matches, output_dir):
    # train predictor for market movements
    
    features = []
    labels = []
    for match in matches:
        # using sentiment and engagement as features - could be better
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
            # simple binary based on engagement for now
            labels.append(1 if match.get('score', 0) > 20 else 0)
    
    if not features:
        print("No training data for market movement")
        return
        
    model = MarketMovementPredictor()
    model.fit(features, labels)
    
    output_path = Path(output_dir) / 'market_movement'
    model.save(output_path)
    
    print(f"Market movement model saved to {output_path}")

def train_bet_sizer(matches, historical_data, output_dir):
    # train model for bet sizing recommendations
    
    features = []
    labels = []
    for match in matches:
        if match.get('sentiment') and match.get('score') is not None:
            feature = {
                'sentiment': match['sentiment'],
                'score': match['score'],
                'num_comments': match.get('num_comments', 0),
                'subreddit': match.get('subreddit', ''),
                'has_url': bool(match.get('url'))
            }
            
            # simple stake sizing based on confidence - needs work
            confidence = match.get('score', 0) / 100.0  # normalize score
            stake = min(0.05, max(0.01, confidence))  # between 1% and 5%
            
            features.append(feature)
            labels.append(stake)
    
    if not features:
        print("No training data for bet sizer")
        return
        
    model = BetSizer()
    model.fit(features, labels)
    
    output_path = Path(output_dir) / 'bet_sizer'
    model.save(output_path)
    
    print(f"Bet sizer model saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Train ML models for betting analysis')
    parser.add_argument('--matches', required=True, help='Path to matches JSON file')
    parser.add_argument('--historical', help='Path to historical data JSON file') 
    parser.add_argument('--output', default='./models', help='Output directory for trained models')
    parser.add_argument('--models', nargs='+', 
                       choices=['team_identifier', 'signal_reliability', 'market_movement', 'bet_sizer', 'all'],
                       default=['all'], help='Which models to train')
    
    args = parser.parse_args()
    
    # load training data
    print(f"Loading training data from {args.matches}")
    matches, historical_data = load_training_data(args.matches, args.historical)
    print(f"Loaded {len(matches)} matches")
    
    # create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # train requested models
    models_to_train = args.models if 'all' not in args.models else ['team_identifier', 'signal_reliability', 'market_movement', 'bet_sizer']
    
    for model_name in models_to_train:
        print(f"\nTraining {model_name}...")
        try:
            if model_name == 'team_identifier':
                train_team_identifier(matches, args.output)
            elif model_name == 'signal_reliability':
                train_signal_reliability(matches, historical_data, args.output)
            elif model_name == 'market_movement':
                train_market_movement(matches, args.output)
            elif model_name == 'bet_sizer':
                train_bet_sizer(matches, historical_data, args.output)
        except Exception as e:
            print(f"Failed to train {model_name}: {str(e)}")
            continue
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main() 