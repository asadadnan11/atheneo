import json
import pandas as pd
from datetime import datetime, timedelta
import os
import glob
import shutil
import functools
import time

class DataHandler:
    def __init__(self):
        self.matches_file = 'data/current_matches.json'
        self.signals_file = 'data/test_pipeline/signals.json'
        self.tweets_file = 'data/test_pipeline/tweets.json'
        self.cache = {}
        self.cache_timeout = 300  # 5 min cache
        
    def cleanup_old_files(self):
        # clean up old files to save space - keep only last 24h
        try:
            now = datetime.now()
            for file_type in ['matches_', 'odds_', 'signals_', 'tweets_']:
                for file in os.listdir('data/history'):
                    if file.startswith(file_type):
                        file_path = os.path.join('data/history', file)
                        file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                        if (now - file_time).days >= 1:  # only keep last day
                            os.remove(file_path)
        except Exception as e:
            print(f"Error cleaning up old files: {str(e)}")
        
    @functools.lru_cache(maxsize=32)
    def get_matches(self, time_range="Last 24 Hours"):
        # get matches with caching - speeds things up
        if not os.path.exists(self.matches_file):
            return pd.DataFrame()
            
        with open(self.matches_file, 'r', encoding='utf-8') as f:
            matches = json.load(f)
            
        df = pd.DataFrame(matches)
        if df.empty:
            return df
            
        # convert timestamp 
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # filter by time range
        now = datetime.now()
        if time_range == "Last Hour":
            cutoff = now - timedelta(hours=1)
        elif time_range == "Last 6 Hours":
            cutoff = now - timedelta(hours=6)
        elif time_range == "Last 24 Hours":
            cutoff = now - timedelta(hours=24)
        else:
            cutoff = now - timedelta(days=7)  # fallback to 7 days
            
        return df[df['timestamp'] >= cutoff]
    
    @functools.lru_cache(maxsize=32)
    def get_team_matches(self, team, time_range="Last 24 Hours"):
        # get matches for specific team
        df = self.get_matches(time_range)
        if df.empty:
            return df
            
        # search in title and text - case insensitive
        return df[df['title'].str.contains(team, case=False) | 
                 df['text'].str.contains(team, case=False)]
    
    @functools.lru_cache(maxsize=32)
    def get_sentiment_stats(self, time_range="Last 24 Hours"):
        # get sentiment breakdown with caching
        df = self.get_matches(time_range)
        if df.empty:
            return {'positive': 0, 'neutral': 0, 'negative': 0}
            
        sentiment_counts = df['sentiment'].value_counts()
        return {
            'positive': sentiment_counts.get('positive', 0),
            'neutral': sentiment_counts.get('neutral', 0),
            'negative': sentiment_counts.get('negative', 0)
        }
    
    @functools.lru_cache(maxsize=32)
    def get_signals(self):
        # load processed signals
        if not os.path.exists(self.signals_file):
            return []
            
        with open(self.signals_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @functools.lru_cache(maxsize=32)
    def get_tweets(self):
        # load tweet summaries
        if not os.path.exists(self.tweets_file):
            return []
            
        with open(self.tweets_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @functools.lru_cache(maxsize=32)
    def get_team_signals(self, team):
        # get signals for specific team only
        signals = self.get_signals()
        return [s for s in signals if s.get('team', '').lower() == team.lower()]
    
    def get_metrics(self):
        # dashboard metrics calculation
        df = self.get_matches()
        signals = self.get_signals()
        
        if df.empty:
            return {
                'total_matches': 0,
                'positive_signals': 0,
                'negative_signals': 0,
                'signal_strength': 0
            }
            
        pos_signals = len([s for s in signals if s.get('signal') == 'positive'])
        neg_signals = len([s for s in signals if s.get('signal') == 'negative'])
        
        return {
            'total_matches': len(df),
            'positive_signals': pos_signals,
            'negative_signals': neg_signals,
            'signal_strength': int((pos_signals + neg_signals) / len(signals) * 100) if signals else 0
        } 