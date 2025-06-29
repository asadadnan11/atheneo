import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer, RobertaForSequenceClassification, RobertaTokenizer
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import Dataset, DataLoader, TensorDataset
from team_aliases import TEAM_ALIASES
import xgboost as xgb
from datetime import datetime
from torch.optim import AdamW
from sklearn.preprocessing import StandardScaler

class TeamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class TeamIdentifier:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def fit(self, texts, teams, epochs=3, batch_size=16):
        # convert to tensors for training
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        labels = torch.tensor(teams)
        
        # setup data loader
        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        
        # training loop - this takes a while
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}')
        
        print("Training completed successfully!")

    def prepare_training_data(self, matches_file):
        # load matches and extract team data
        with open(matches_file, 'r', encoding='utf-8') as f:
            matches = json.load(f)
        
        texts = []
        labels = []
        
        for match in matches:
            text = f"{match.get('title', '')} {match.get('text', '')}"
            if not text.strip():
                continue
                
            team = match.get('inferred_team')
            if team and team in TEAM_ALIASES:
                texts.append(text)
                labels.append(TEAM_ALIASES[team])
        
        return texts, labels
    
    def predict(self, text):
        # get predictions for text input
        self.model.eval()
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        probs = torch.softmax(outputs.logits, dim=1)
        team_probs = {
            team: prob.item()
            for team, prob in zip(TEAM_ALIASES.keys(), probs[0])
        }
        
        return team_probs
    
    def save_model(self, path):
        # save trained model
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

class SignalReliability(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def fit(self, features, labels, epochs=3, batch_size=16):
        # process text content
        texts = [f['content'] for f in features]
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        
        # handle numerical features
        numerical = torch.tensor([[
            f['score'],
            f['num_comments']
        ] for f in features], dtype=torch.float32)
        
        # scale the numerical stuff
        numerical = torch.tensor(self.scaler.fit_transform(numerical), dtype=torch.float32)
        
        labels = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)
        
        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], numerical, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = AdamW(self.parameters(), lr=2e-5)
        criterion = nn.MSELoss()
        
        # train the model
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                numerical = batch[2].to(self.device)
                labels = batch[3].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask).logits
                outputs = torch.cat([outputs, numerical], dim=1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}')
    
    def save_pretrained(self, path):
        # save model state
        torch.save(self.state_dict(), f"{path}/model.pt")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def extract_features(self, post):
        # extract features from post data - lots of manual feature engineering here
        features = []
        
        # basic stats
        features.append(post.get('score', 0))
        features.append(post.get('num_comments', 0))
        features.append(len(post.get('title', '')))
        features.append(len(post.get('text', '')))
        
        # time features
        timestamp = post.get('timestamp')
        if timestamp:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            features.append(dt.hour)  # hour of day
            features.append(dt.weekday())  # day of week
        else:
            features.extend([0, 0])
        
        # text features - pretty basic stuff
        title = post.get('title', '').lower()
        text = post.get('text', '').lower()
        combined_text = f"{title} {text}"
        
        # keyword indicators
        betting_keywords = ['bet', 'odds', 'tip', 'prediction', 'stake']
        features.append(sum(1 for word in betting_keywords if word in combined_text))
        
        # sentiment indicators - rough approximation
        positive_words = ['good', 'great', 'excellent', 'strong', 'confident']
        negative_words = ['bad', 'terrible', 'avoid', 'risky', 'dangerous']
        features.append(sum(1 for word in positive_words if word in combined_text))
        features.append(sum(1 for word in negative_words if word in combined_text))
        
        return np.array(features)
    
    def _extract_categories(self, post):
        # categorize post content - could be smarter but this works
        categories = []
        text = f"{post.get('title', '')} {post.get('text', '')}".lower()
        
        if any(word in text for word in ['injury', 'injured', 'hurt', 'out']):
            categories.append('injury')
        if any(word in text for word in ['lineup', 'starting', 'formation']):
            categories.append('lineup')
        if any(word in text for word in ['form', 'streak', 'run']):
            categories.append('form')
        if any(word in text for word in ['weather', 'rain', 'wind', 'conditions']):
            categories.append('weather')
        if any(word in text for word in ['odds', 'betting', 'bookmaker']):
            categories.append('betting_market')
            
        return categories
    
    def predict(self, post):
        # predict reliability score for a post
        self.eval()
        features = self.extract_features(post)
        
        with torch.no_grad():
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            score = self.forward(features_tensor).item()
        
        # get category breakdown too
        categories = self._extract_categories(post)
        category_scores = {cat: 0.8 if cat in categories else 0.2 for cat in ['injury', 'lineup', 'form', 'weather', 'betting_market']}
        
        return score, category_scores
    
    def update_historical_data(self, user_id, was_correct):
        # track user reliability over time - simple running average for now
        # TODO: implement proper user tracking
        pass

class MarketMovementPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def fit(self, features: List[Dict], labels: List[int], epochs: int = 100, batch_size: int = 32):
        try:
            # Convert features to tensor
            X = torch.tensor([[
                f['sentiment'],
                f['score'],
                f['num_comments']
            ] for f in features], dtype=torch.float32)
            
            # Scale features
            X = torch.tensor(self.scaler.fit_transform(X), dtype=torch.float32)
            
            # Convert labels
            y = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)
            
            # Create dataset
            dataset = TensorDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Setup optimizer
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
            criterion = nn.BCELoss()
            
            # Training loop
            self.train()
            for epoch in range(epochs):
                total_loss = 0
                for batch_X, batch_y in dataloader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}')
            
            print("Training completed successfully!")
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise

    def save(self, path: str):
        """Save the model and scaler"""
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_state': self.scaler
        }, str(Path(path) / 'model.pt'))

    def load(self, path: str):
        """Load the model and scaler"""
        checkpoint = torch.load(str(Path(path) / 'model.pt'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler_state']

    def predict_movement(self, 
                        historical_odds: List[float],
                        volume_data: List[float],
                        time_data: List[datetime]) -> Dict[str, float]:
        """Predict probability of significant market movement"""
        # Prepare input features
        odds_changes = np.diff(historical_odds)
        volume_changes = np.diff(volume_data)
        time_diffs = [(t - time_data[0]).total_seconds() for t in time_data[1:]]
        
        # Normalize features
        odds_changes = (odds_changes - np.mean(odds_changes)) / np.std(odds_changes)
        volume_changes = (volume_changes - np.mean(volume_changes)) / np.std(volume_changes)
        time_diffs = (time_diffs - np.mean(time_diffs)) / np.std(time_diffs)
        
        # Combine features
        features = np.column_stack([odds_changes, volume_changes, time_diffs])
        features = torch.FloatTensor(features).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            probability = self(features).item()
        
        return {
            'movement_probability': probability,
            'odds_change': odds_changes[-1],
            'volume_change': volume_changes[-1]
        }

class BetSizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def fit(self, features: List[Dict], labels: List[float], epochs: int = 3, batch_size: int = 16):
        try:
            # Process text features
            texts = [f"{f['subreddit']} {f['sentiment']}" for f in features]
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            
            # Process numerical features
            numerical = torch.tensor([[
                f['score'],
                f['num_comments'],
                float(f['has_url'])
            ] for f in features], dtype=torch.float32)
            
            # Scale numerical features
            numerical = torch.tensor(self.scaler.fit_transform(numerical), dtype=torch.float32)
            
            # Convert labels
            labels = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)
            
            # Create dataset
            dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], numerical, labels)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Setup optimizer
            optimizer = AdamW(self.parameters(), lr=2e-5)
            criterion = nn.MSELoss()
            
            # Training loop
            self.train()
            for epoch in range(epochs):
                total_loss = 0
                for batch in dataloader:
                    input_ids = batch[0].to(self.device)
                    attention_mask = batch[1].to(self.device)
                    numerical = batch[2].to(self.device)
                    labels = batch[3].to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(input_ids, attention_mask=attention_mask).logits
                    outputs = torch.cat([outputs, numerical], dim=1)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}')
            
            print("Training completed successfully!")
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise

    def save(self, path: str):
        """Save the model, tokenizer, and scaler"""
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        torch.save(self.scaler, str(Path(path) / 'scaler.pt'))

    def load(self, path: str):
        """Load the model, tokenizer, and scaler"""
        self.model = BertForSequenceClassification.from_pretrained(path)
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.scaler = torch.load(str(Path(path) / 'scaler.pt'))

    def extract_features(self, 
                        signal: Dict,
                        odds: float,
                        bankroll: float,
                        risk_tolerance: float) -> np.ndarray:
        """Extract features for bet sizing"""
        features = []
        
        # Signal features
        features.append(signal.get('reliability_score', 0.5))
        features.append(signal.get('confidence', 0.5))
        
        # Odds features
        features.append(odds)
        features.append(1/odds)  # Implied probability
        
        # Bankroll features
        features.append(bankroll)
        features.append(risk_tolerance)
        
        # Historical performance
        features.append(signal.get('historical_win_rate', 0.5))
        features.append(signal.get('historical_roi', 0.0))
        
        return np.array(features).reshape(1, -1)
    
    def calculate_stake(self,
                       signal: Dict,
                       odds: float,
                       bankroll: float,
                       risk_tolerance: float) -> float:
        """Calculate optimal stake size"""
        features = self.extract_features(signal, odds, bankroll, risk_tolerance)
        stake_fraction = self.model.predict(features)[0][0]
        
        # Ensure stake is within reasonable bounds
        stake_fraction = np.clip(stake_fraction, 0.01, 0.1)
        return stake_fraction * bankroll 