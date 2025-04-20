import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
import os
import functools
import re
import numpy as np

# Page config
st.set_page_config(
    page_title="WhisperBet Insights",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - optimized
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stSelectbox>div>div>select {
        background-color: #1E1E1E;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #1E1E1E;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

class DataLoader:
    def __init__(self):
        self.data_dir = Path("data")
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes cache timeout
        
    @functools.lru_cache(maxsize=32)
    def get_latest_file(self, pattern: str) -> Path:
        """Get the most recent file matching the pattern with caching."""
        files = sorted(self.data_dir.glob(pattern))
        return files[-1] if files else None
    
    @functools.lru_cache(maxsize=32)
    def load_json(self, file_path: Path) -> List[Dict]:
        """Load JSON data from file with caching."""
        if not file_path or not file_path.exists():
            return []
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @functools.lru_cache(maxsize=32)
    def get_matches(self) -> List[Dict]:
        """Get latest matches data with caching."""
        file_path = self.get_latest_file("matches_*.json")
        return self.load_json(file_path)
    
    @functools.lru_cache(maxsize=32)
    def get_signals(self) -> List[Dict]:
        """Get latest signals data with caching."""
        file_path = self.get_latest_file("gpt_insights_*.json")
        return self.load_json(file_path)
    
    @functools.lru_cache(maxsize=32)
    def get_tweets(self) -> List[Dict]:
        """Get latest tweets data with caching."""
        file_path = self.get_latest_file("alpha_bets_tweets_*.json")
        if not file_path:
            return []
        try:
            tweets = self.load_json(file_path)
            # Convert list to dict if needed
            if isinstance(tweets, list):
                return tweets
            # If it's a dict with a 'tweets' key, return that
            if isinstance(tweets, dict) and 'tweets' in tweets:
                return tweets['tweets']
            return []
        except Exception as e:
            print(f"Error loading tweets: {e}")
            return []

# Cache expensive computations
@st.cache_data(ttl=300)
def format_match_data(matches: List[Dict]) -> pd.DataFrame:
    """Format match data with caching"""
    if not matches:
        return pd.DataFrame()
        
    data = []
    for match in matches:
        if 'title' in match:
            data.append({
                'Title': match['title'],
                'Subreddit': match['subreddit'],
                'Sentiment': match.get('sentiment', 'neutral'),
                'Score': match.get('score', 0),
                'Comments': match.get('num_comments', 0),
                'Time': datetime.fromisoformat(match['timestamp'].replace('Z', '+00:00'))
            })
    
    return pd.DataFrame(data)

def analyze_betting_sentiment(tweet_text: str) -> tuple:
    """Analyze betting sentiment from tweet text with betting-specific language patterns."""
    if not tweet_text:
        return 'neutral', 0.0
        
    tweet_text = tweet_text.lower()
    
    # Base confidence from explicit statements
    confidence_score = 0.4  # Default moderate confidence
    if 'confidence: high' in tweet_text:
        confidence_score = 0.7
    elif 'confidence: medium' in tweet_text:
        confidence_score = 0.5
    elif 'confidence: low' in tweet_text:
        confidence_score = 0.3

    # Strong positive signals (clear value + confidence)
    strong_value_patterns = [
        r'max (?:bet|play|unit)',
        r'(?:top|best) (?:bet|play)',
        r'strong value',
        r'excellent (?:value|odds)',
        r'high confidence.*value',
        r'hammer this',
        r'cant miss',
        r'lock'
    ]
    
    # Positive signals (value or edge identified)
    value_patterns = [
        r'value (?:bet|play|pick)',
        r'good (?:odds|price)',
        r'worth (?:a|the) (?:look|play)',
        r'smart play',
        r'favorable',
        r'advantage',
        r'lean(?:ing)? \w+'
    ]
    
    # Neutral/cautious signals (uncertainty or small stakes)
    neutral_patterns = [
        r'consider',
        r'could go either way',
        r'(?:50|fifty)[ -](?:50|fifty)',
        r'even match',
        r'wait and see',
        r'small (?:bet|play|stake)',
        r'slight lean',
        r'keep an eye',
        r'might be',
        r'(?:tight|close) odds',
        r'upset (?:potential|alert)',
        r'dont sleep on'
    ]
    
    # Negative signals (warnings or poor value)
    negative_patterns = [
        r'avoid',
        r'stay away',
        r'skip',
        r'pass',
        r'trap',
        r'too (?:risky|expensive)',
        r'dangerous',
        r'bad (?:odds|value)',
        r'poor value',
        r'no value',
        r'high risk',
        r'sharp action against'
    ]

    # Count matches
    strong_value_count = sum(1 for p in strong_value_patterns if re.search(p, tweet_text))
    value_count = sum(1 for p in value_patterns if re.search(p, tweet_text))
    neutral_count = sum(1 for p in neutral_patterns if re.search(p, tweet_text))
    negative_count = sum(1 for p in negative_patterns if re.search(p, tweet_text))
    
    # Extract and analyze odds
    odds_pattern = r'(?:odds?[:)]?\s*(?:at|of)?\s*)?(\d+(?:\.\d+)?)'
    odds_matches = re.findall(odds_pattern, tweet_text)
    odds_values = [float(o) for o in odds_matches if float(o) > 1.0]
    
    # Odds-based sentiment
    odds_sentiment = 0
    if odds_values:
        avg_odds = sum(odds_values) / len(odds_values)
        if avg_odds > 3.0:  # High odds
            if any(p in tweet_text for p in ['value', 'worth', 'good']):
                odds_sentiment = 0.2  # Slight positive for value on underdog
            else:
                odds_sentiment = -0.1  # Slight negative for high risk
        elif avg_odds < 1.5:  # Low odds
            if any(p in tweet_text for p in ['trap', 'risky', 'juice']):
                odds_sentiment = -0.2  # Negative for potential traps
            else:
                odds_sentiment = 0.1  # Slight positive for favorites

    # Calculate base score with betting-specific weights
    base_score = (
        strong_value_count * 0.6 +     # Strong conviction but reduced impact
        value_count * 0.3 +            # Moderate positive for value found
        neutral_count * -0.15 +        # Slight negative for uncertainty
        negative_count * -0.5 +        # Strong negative but not overwhelming
        odds_sentiment                 # Slight odds adjustment
    )
    
    # Apply confidence multiplier
    sentiment_score = base_score * confidence_score
    
    # Thresholds calibrated to betting language
    if sentiment_score > 0.5:  # Very high conviction needed
        return 'strongly positive', sentiment_score
    elif sentiment_score > 0.15:  # Lower threshold for positive
        return 'positive', sentiment_score
    elif sentiment_score > -0.15:  # Wide neutral band
        return 'neutral', sentiment_score
    elif sentiment_score > -0.5:  # Significant negative signal needed
        return 'negative', sentiment_score
    else:
        return 'strongly negative', sentiment_score

@st.cache_data(ttl=300)
def plot_sentiment_distribution(signals: List[Dict]):
    """Plot sentiment distribution with caching."""
    if not signals:
        st.warning("No signals data available")
        return
        
    # Extract sentiment from betting recommendations
    sentiments = []
    confidence_scores = []
    
    for s in signals:
        tweet = s.get('tweet', '')
        if tweet:
            sentiment, score = analyze_betting_sentiment(tweet)
            sentiments.append(sentiment)
            confidence_scores.append(abs(score))
    
    if not sentiments:
        st.warning("No sentiment data available")
        return
    
    # Calculate average confidence
    avg_confidence = sum(confidence_scores) / len(confidence_scores)
    st.metric("Average Confidence Score", f"{avg_confidence:.2f}")
    
    # Create sentiment distribution
    sentiment_counts = pd.Series(sentiments).value_counts()
    
    # Define sentiment order and colors
    sentiment_order = ['strongly positive', 'positive', 'neutral', 'negative', 'strongly negative']
    colors = {
        'strongly positive': '#2E7D32',
        'positive': '#66BB6A',
        'neutral': '#90A4AE',
        'negative': '#EF5350',
        'strongly negative': '#C62828'
    }
    
    # Reindex to ensure all categories are present
    sentiment_counts = sentiment_counts.reindex(sentiment_order, fill_value=0)
    
    # Calculate percentages
    total = sentiment_counts.sum()
    percentages = (sentiment_counts / total * 100).round(1)
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        y=sentiment_counts.index,
        x=percentages,
        orientation='h',
        marker_color=[colors[s] for s in sentiment_counts.index],
        text=[f"{p:.1f}%" for p in percentages],
        textposition='auto',
        hovertemplate="<b>%{y}</b><br>" +
                     "Percentage: %{x:.1f}%<br>" +
                     "Count: %{text}<extra></extra>"
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Betting Sentiment Distribution',
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(size=16)
        ),
        xaxis_title="Percentage of Signals",
        yaxis_title=None,
        yaxis=dict(
            categoryorder='array',
            categoryarray=sentiment_order[::-1],  # Reverse order for better visual
            tickfont=dict(size=12)
        ),
        xaxis=dict(
            ticksuffix="%",
            range=[0, max(percentages) * 1.1],  # Add 10% padding
            tickfont=dict(size=12)
        ),
        margin=dict(t=80, l=20, r=20, b=60),
        height=400,
        showlegend=False
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def create_metrics_row(data: List[Dict], title: str, metric: str):
    """Create a row of metrics."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total " + title, len(data))
    with col2:
        # Count active signals (upcoming matches within next 24 hours)
        active = 0
        now = datetime.now(timezone.utc)
        
        for d in data:
            try:
                # For tweets/signals, use match_time to determine if active
                if 'match_time' in d:
                    match_time = datetime.fromisoformat(d['match_time'].replace('Z', '+00:00')).astimezone(timezone.utc)
                    # Signal is active if match is within next 24 hours
                    time_diff = (match_time - now).total_seconds()
                    if 0 <= time_diff <= 86400:
                        active += 1
                # For regular posts, use timestamp/created_at
                else:
                    for ts_field in ['timestamp', 'created_at']:
                        ts = d.get(ts_field)
                        if ts:
                            if isinstance(ts, (int, float)):
                                post_time = datetime.fromtimestamp(ts, timezone.utc)
                            else:
                                post_time = datetime.fromisoformat(ts.replace('Z', '+00:00')).astimezone(timezone.utc)
                            
                            time_diff = (now - post_time).total_seconds()
                            if time_diff < 86400:
                                active += 1
                            break
            except (ValueError, KeyError) as e:
                print(f"Error processing date: {e}")
                continue
                
        metric_label = "Active Matches" if title == "Signals" else "Active " + title
        st.metric(metric_label, active)
    with col3:
        if metric in ["sentiment", "score"]:
            # Calculate sentiment scores using the new analysis
            scores = []
            for d in data:
                tweet = d.get('tweet', '')
                if tweet:
                    _, score = analyze_betting_sentiment(tweet)
                    scores.append(score)
            
            avg = sum(scores) / len(scores) if scores else 0
            st.metric(f"Avg Confidence", f"{avg:.2f}")
        else:
            st.metric(f"Posts Today", active)

# Common team name patterns with variations
team_patterns = {
    'Arsenal': ['\\barsenal\\b', '\\bafc\\b', '\\bgunners\\b', '\\bthe\\s+gunners\\b'],
    'Chelsea': ['\\bchelsea\\b', '\\bcfc\\b', '\\bthe\\s+blues\\b', '\\bchels\\b'],
    'Manchester City': ['\\bman\\s*city\\b', '\\bmcfc\\b', '\\bcity\\b(?!\\s+vs|\\.)', '\\bthe\\s+citizens\\b'],
    'Liverpool': ['\\bliverpool\\b', '\\blfc\\b', '\\bthe\\s+reds\\b', '\\bpool\\b'],
    'Manchester United': ['\\bman\\s*u(?:td|nited)?\\b', '\\bmufc\\b', '\\bunited\\b(?!\\s+vs|\\.)', '\\bred\\s+devils\\b'],
    'Tottenham': ['\\btottenham\\b', '\\bspurs\\b', '\\bthfc\\b', '\\bhotspur\\b', '\\bcoys\\b'],
    'Fulham': ['\\bfulham\\b', '\\bffc\\b', '\\bthe\\s+cottagers\\b'],
    'Wolves': ['\\bwolves\\b', '\\bwolverhampton\\b', '\\bwwfc\\b'],
    'Newcastle': ['\\bnewcastle\\b', '\\bnufc\\b', '\\bthe\\s+magpies\\b', '\\btoon\\b'],
    'Brighton': ['\\bbrighton\\b', '\\bbhafc\\b', '\\bhove\\b', '\\bthe\\s+seagulls\\b'],
    'Aston Villa': ['\\baston\\s*villa\\b', '\\bavfc\\b', '\\bvilla\\b(?!\\s+vs|\\.)', '\\bthe\\s+villans\\b'],
    'West Ham': ['\\bwest\\s*ham\\b', '\\bwhufc\\b', '\\bthe\\s+hammers\\b'],
    'Crystal Palace': ['\\bcrystal\\s*palace\\b', '\\bcpfc\\b', '\\bpalace\\b(?!\\s+vs|\\.)', '\\bthe\\s+eagles\\b'],
    'Everton': ['\\beverton\\b', '\\befc\\b', '\\bthe\\s+toffees\\b'],
    'Burnley': ['\\bburnley\\b', '\\bbfc\\b', '\\bthe\\s+clarets\\b'],
    'Nottingham Forest': ['\\bnottingham\\s*forest\\b', '\\bnffc\\b', '\\bforest\\b(?!\\s+vs|\\.)', '\\bthe\\s+tricky\\s+trees\\b'],
    'Brentford': ['\\bbrentford\\b', '\\bbees\\b(?!\\s+vs|\\.)', '\\bthe\\s+bees\\b'],
    'Luton': ['\\bluton\\b', '\\bltfc\\b', '\\bthe\\s+hatters\\b'],
    'Ipswich Town': ['\\bipswich\\b', '\\bitfc\\b', '\\bthe\\s+tractor\\s+boys\\b']
}

@st.cache_data(ttl=300)  # Cache for 5 minutes
def plot_team_analysis(matches: List[Dict], tweets: List[Dict] = None):
    """Plot team analysis with caching."""
    if not matches and not tweets:
        st.warning("No data available")
        return
        
    teams = []
    
    # Process matches
    if matches:
        for match in matches:
            # Extract teams from title and text
            title = match.get('title', '').lower()
            text = match.get('text', '').lower()
            
            # Look for team patterns
            for team_name, patterns in team_patterns.items():
                if any(re.search(pattern, f" {title} {text} ", re.IGNORECASE) for pattern in patterns):
                    teams.append(team_name)
    
    # Process tweets
    if tweets:
        for tweet in tweets:
            # Add teams from structured data
            home_team = tweet.get('home_team')
            away_team = tweet.get('away_team')
            if home_team:
                teams.append(home_team)
            if away_team:
                teams.append(away_team)
            
            # Also check tweet text for additional mentions
            tweet_text = tweet.get('tweet', '').lower()
            for team_name, patterns in team_patterns.items():
                if any(re.search(pattern, f" {tweet_text} ", re.IGNORECASE) for pattern in patterns):
                    teams.append(team_name)
    
    if not teams:
        st.warning("No team data available")
        return
    
    # Count mentions and create chart
    team_counts = pd.Series(teams).value_counts()
    
    fig = px.bar(
        x=team_counts.index,
        y=team_counts.values,
        title='Teams by Mention Count',
        labels={'x': 'Team', 'y': 'Number of Mentions'},
        color_discrete_sequence=['#1E88E5']
    )
    
    fig.update_layout(
        height=500,
        margin=dict(t=100, l=20, r=20, b=180),  # Further increase bottom margin
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_tweets(tweets: List[Dict]):
    """Display the latest tweets with proper formatting."""
    if not tweets:
        st.info("No tweets available")
        return
        
    st.subheader("Latest Betting Tips 🎯")
    
    for tweet in tweets[:5]:  # Display latest 5 tweets
        try:
            # Format timestamps
            match_time = datetime.fromisoformat(tweet['match_time'].replace('Z', '+00:00'))
            generated_at = datetime.fromisoformat(tweet['generated_at'].replace('Z', '+00:00'))
            
            # Create styled HTML for the tweet
            tweet_html = f"""
            <div style='background-color: #1E1E1E; padding: 15px; border-radius: 10px; margin: 10px 0;'>
                <div style='color: #1E88E5; margin-bottom: 5px;'>
                    {tweet['home_team']} vs {tweet['away_team']}
                </div>
                <div style='color: #FFFFFF; margin: 10px 0;'>
                    {tweet['tweet']}
                </div>
                <div style='color: #888888; font-size: 0.8em;'>
                    Match Time: {match_time.strftime('%Y-%m-%d %H:%M UTC')}
                    <br>
                    Generated: {generated_at.strftime('%Y-%m-%d %H:%M UTC')}
                </div>
            </div>
            """
            st.markdown(tweet_html, unsafe_allow_html=True)
            
        except (KeyError, ValueError) as e:
            st.error(f"Error displaying tweet: {str(e)}")
            continue

def main():
    st.title("WhisperBet Insights 🎲")
    
    # Initialize data loader
    data_loader = DataLoader()
    
    # Load data
    matches = data_loader.get_matches()
    signals = data_loader.get_signals()
    tweets = data_loader.get_tweets()
    
    # Combine signals and tweets for sentiment analysis
    all_signals = []
    if signals:
        all_signals.extend(signals)
    if tweets:
        all_signals.extend(tweets)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Overview 📊", "Analysis 📈", "Tweets 🐦"])
    
    with tab1:
        st.header("Overview")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            create_metrics_row(matches, "Posts", "posts")
            create_metrics_row(all_signals, "Signals", "sentiment")  # Changed from signals to all_signals
            
        with col2:
            plot_sentiment_distribution(all_signals)  # Changed from signals to all_signals
            
    with tab2:
        st.header("Team Analysis")
        plot_team_analysis(matches, tweets)
        
    with tab3:
        display_tweets(tweets)

if __name__ == "__main__":
    main() 