import os
# type: ignore
from dotenv import load_dotenv
load_dotenv()

print("ENV loaded:")
print("CLIENT_ID =", os.getenv("REDDIT_CLIENT_ID"))
print("CLIENT_SECRET =", os.getenv("REDDIT_CLIENT_SECRET"))
print("USER_AGENT =", os.getenv("REDDIT_USER_AGENT"))

import praw
import json
import os
from datetime import datetime, timedelta
import time
import logging
import argparse


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Reddit API credentials
CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USER_AGENT = os.getenv("REDDIT_USER_AGENT")

# Ensure data directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('data/test_pipeline', exist_ok=True)
os.makedirs('data/history', exist_ok=True)  # New directory for historical files

# Subreddits to monitor - reduced to essential ones
SUBREDDITS = [
    # Premier League
    'PremierLeague', 'Gunners', 'MCFC', 'LiverpoolFC', 'chelseafc',
    'coys', 'reddevils',
    
    # Other Leagues
    'soccer', 'LaLiga', 'Barca', 'realmadrid',
    
    # Betting & Analysis
    'soccerbetting', 'soccernerd'
]

# Keywords to track
KEYWORDS = [
    # Injury & Fitness
    'injury', 'injured', 'fitness', 'recovery', 'knock', 'strain', 'hamstring',
    'muscle', 'concussion', 'medical', 'scan', 'rehab', 'physio', 'treatment',
    
    # Team News
    'lineup', 'starting xi', 'team news', 'squad', 'selection', 'rotation',
    'rested', 'dropped', 'bench', 'substitute', 'reserve', 'suspended',
    'ban', 'red card', 'yellow card', 'accumulated',
    
    # Form & Performance
    'form', 'momentum', 'confidence', 'morale', 'atmosphere', 'pressure',
    'struggling', 'firing', 'peaking', 'slump', 'crisis', 'rebound',
    
    # Tactical
    'tactics', 'formation', 'system', 'style', 'approach', 'gameplan',
    'strategy', 'setup', 'shape', 'press', 'counter', 'possession',
    
    # Manager & Staff
    'coach', 'manager', 'head coach', 'boss', 'tactician', 'staff',
    'training', 'preparation', 'analysis', 'scout', 'scouting',
    
    # Transfer & Squad
    'transfer', 'signing', 'deal', 'contract', 'negotiation', 'agreement',
    'squad', 'roster', 'depth', 'cover', 'option', 'alternative',
    
    # Weather & Conditions
    'weather', 'pitch', 'condition', 'surface', 'grass', 'wet', 'dry',
    'wind', 'rain', 'snow', 'temperature', 'climate', 'stadium',
    
    # Schedule & Fixtures
    'schedule', 'fixture', 'congestion', 'rotation', 'rest', 'recovery',
    'travel', 'away', 'home', 'stadium', 'venue', 'location',
    
    # Betting Specific
    'odds', 'value', 'edge', 'bet', 'wager', 'stake', 'bankroll',
    'handicap', 'spread', 'over/under', 'total', 'moneyline',
    'accumulator', 'parlay', 'treble', 'double', 'single',
    'bookmaker', 'bookie', 'sharp', 'square', 'public money',
    'line movement', 'odds movement', 'price', 'juice', 'vig',
    'favorite', 'underdog', 'pick', 'tip', 'prediction'
]

# Sentiment Keywords - optimized for performance
SENTIMENT_KEYWORDS = {
    'positive': [
        'boost', 'return', 'available', 'fit', 'ready', 'sharp',
        'confident', 'motivated', 'determined', 'focused',
        'value', 'edge', 'sharp', 'lock', 'sure', 'guaranteed',
        'free money', 'easy money', 'sure thing', 'banker'
    ],
    'negative': [
        'doubt', 'concern', 'worry', 'issue', 'problem', 'struggle',
        'fatigue', 'tired', 'exhausted', 'doubtful',
        'trap', 'fade', 'avoid', 'stay away', 'dangerous',
        'risky', 'uncertain', 'volatile', 'unpredictable'
    ]
}

# Output file path
OUTPUT_FILE = 'data/reddit_stream.json'

# Global storage for matches
class MatchStorage:
    def __init__(self):
        self.matches = []
        self.matches_file = 'data/current_matches.json'
        self.history_dir = 'data/history'
    
    def clear(self):
        """Clear the current matches list"""
        self.matches = []
    
    def add_matches(self, new_matches):
        """Add new matches to the list"""
        if isinstance(new_matches, list):
            self.matches.extend(new_matches)
        else:
            self.matches.append(new_matches)
    
    def get_matches(self):
        """Get the current matches list"""
        return self.matches
    
    def cleanup_old_files(self):
        """Clean up old history files, keeping only the last 7 days"""
        try:
            for file in os.listdir(self.history_dir):
                if file.startswith('matches_'):
                    file_path = os.path.join(self.history_dir, file)
                    file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                    if (datetime.now() - file_time).days > 7:
                        os.remove(file_path)
        except Exception as e:
            logging.error(f"Error cleaning up old files: {str(e)}")
    
    def save_matches(self):
        try:
            # Ensure matches is a list
            if not isinstance(self.matches, list):
                self.matches = list(self.matches)
            
            # Validate each match
            validated_matches = []
            for match in self.matches:
                if isinstance(match, dict):
                    validated_match = {}
                    for key, value in match.items():
                        if isinstance(value, str):
                            validated_match[key] = value.encode('utf-8', errors='ignore').decode('utf-8')
                        else:
                            validated_match[key] = value
                    validated_matches.append(validated_match)
            
            # Save current matches
            with open(self.matches_file, 'w', encoding='utf-8') as f:
                json.dump(validated_matches, f, indent=2, ensure_ascii=False)
            logging.info(f"Saved {len(validated_matches)} matches to {self.matches_file}")
            
            # Save historical copy
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            history_file = os.path.join(self.history_dir, f"matches_{timestamp}.json")
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(validated_matches, f, indent=2, ensure_ascii=False)
            logging.info(f"Saved historical copy to {history_file}")
            
            # Cleanup old files
            self.cleanup_old_files()
            
        except Exception as e:
            logging.error(f"Error saving matches: {str(e)}")
            # Create an empty matches file if saving fails
            with open(self.matches_file, 'w', encoding='utf-8') as f:
                json.dump([], f)

# Create global storage instance
match_storage = MatchStorage()

def init_reddit():
    """Initialize Reddit API client"""
    return praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT
    )

def analyze_sentiment(text):
    """Analyze sentiment of text based on keywords"""
    text_lower = text.lower()
    positive_count = sum(1 for word in SENTIMENT_KEYWORDS['positive'] if word in text_lower)
    negative_count = sum(1 for word in SENTIMENT_KEYWORDS['negative'] if word in text_lower)
    
    if positive_count > negative_count:
        return 'positive'
    elif negative_count > positive_count:
        return 'negative'
    return 'neutral'

def check_keywords(text):
    """Check if any keywords appear in the text"""
    return any(keyword.lower() in text.lower() for keyword in KEYWORDS)

def monitor_subreddits(single_cycle=False):
    """Monitor subreddits for posts matching keywords"""
    reddit = init_reddit()
    start_time = time.time()
    max_duration = 60  # Reduced from 120 to 60 seconds
    
    while True:
        logging.info("Starting new monitoring cycle...")
        match_storage.clear()
        
        for subreddit in SUBREDDITS:
            try:
                # Reduced limit from 5 to 3 posts per subreddit
                for post in reddit.subreddit(subreddit).new(limit=3):
                    if check_keywords(post.title) or check_keywords(post.selftext):
                        sentiment = analyze_sentiment(f"{post.title} {post.selftext}")
                        match_storage.add_matches([{
                            'subreddit': subreddit,
                            'title': post.title,
                            'text': post.selftext,
                            'url': post.url,
                            'created_utc': post.created_utc,
                            'timestamp': datetime.now().isoformat(),
                            'sentiment': sentiment,
                            'score': post.score,
                            'num_comments': post.num_comments
                        }])
                        
                    try:
                        post.comments.replace_more(limit=0)
                        # Only process top 5 comments instead of all
                        for comment in post.comments.list()[:5]:
                            if check_keywords(comment.body):
                                sentiment = analyze_sentiment(comment.body)
                                match_storage.add_matches([{
                                    'subreddit': subreddit,
                                    'parent_title': post.title,
                                    'comment_text': comment.body,
                                    'url': comment.permalink,
                                    'created_utc': comment.created_utc,
                                    'timestamp': datetime.now().isoformat(),
                                    'sentiment': sentiment,
                                    'score': comment.score
                                }])
                    except Exception as e:
                        logging.error(f"Error processing comments for post: {str(e)}")
                        continue
                    
            except Exception as e:
                logging.error(f"Error monitoring r/{subreddit}: {str(e)}")
                continue
                
        if match_storage.get_matches():
            logging.info(f"Found {len(match_storage.get_matches())} new matches")
            
            # Also save to timestamped file
            filename = f"data/matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(match_storage.get_matches(), f, indent=2)
            logging.info(f"Saved matches to {filename}")

        if single_cycle or (time.time() - start_time) >= max_duration:
            logging.info("Reached maximum duration of 1 minute. Stopping harvester.")
            break
            
        time.sleep(30)  # Increased from 10 to 30 seconds

def collect_overnight_data():
    """Collect data from the past 8 hours"""
    reddit = init_reddit()
    cutoff_time = datetime.utcnow() - timedelta(hours=8)
    total_subs = len(SUBREDDITS)
    
    logging.info(f"Starting overnight data collection for {total_subs} subreddits...")
    
    for idx, subreddit in enumerate(SUBREDDITS, 1):
        try:
            logging.info(f"Processing subreddit {idx}/{total_subs}: r/{subreddit}")
            subreddit_posts = []
            
            for post in reddit.subreddit(subreddit).new(limit=500):
                post_time = datetime.utcfromtimestamp(post.created_utc)
                if post_time < cutoff_time:
                    break
                    
                if check_keywords(post.title) or check_keywords(post.selftext):
                    sentiment = analyze_sentiment(f"{post.title} {post.selftext}")
                    subreddit_posts.append({
                        'subreddit': subreddit,
                        'title': post.title,
                        'text': post.selftext,
                        'url': post.url,
                        'created_utc': post.created_utc,
                        'timestamp': datetime.now().isoformat(),
                        'sentiment': sentiment,
                        'score': post.score,
                        'num_comments': post.num_comments
                    })
            
            for post in subreddit_posts:
                try:
                    post_obj = reddit.submission(url=post['url'])
                    post_obj.comments.replace_more(limit=0)
                    
                    for comment in post_obj.comments.list():
                        if check_keywords(comment.body):
                            sentiment = analyze_sentiment(comment.body)
                            match_storage.add_matches([{
                                'subreddit': subreddit,
                                'parent_title': post['title'],
                                'comment_text': comment.body,
                                'url': comment.permalink,
                                'created_utc': comment.created_utc,
                                'timestamp': datetime.now().isoformat(),
                                'sentiment': sentiment,
                                'score': comment.score
                            }])
                except Exception as e:
                    logging.error(f"Error processing comments for post: {str(e)}")
                    continue
            
            match_storage.add_matches(subreddit_posts)
            logging.info(f"Found {len(subreddit_posts)} matches in r/{subreddit}")
            
        except Exception as e:
            logging.error(f"Error monitoring r/{subreddit}: {str(e)}")
            continue
            
    logging.info(f"Overnight collection complete. Total matches: {len(match_storage.get_matches())}")
    return match_storage.get_matches()

def main():
    parser = argparse.ArgumentParser(description='Reddit Sports Betting Insights Monitor')
    parser.add_argument('--timeframe', choices=['live', 'overnight'], default='live',
                      help='Timeframe for data collection')
    parser.add_argument('--output', type=str, default=None,
                      help='Output file path for collected data')
    parser.add_argument('--limit', type=int, default=500,
                      help='Maximum number of posts to process per subreddit')
    parser.add_argument('--single', action='store_true',
                      help='Run only a single cycle instead of continuous monitoring')
    args = parser.parse_args()
    
    if args.timeframe == 'overnight':
        start_time = time.time()
        logging.info("Starting overnight data collection...")
        
        matches = collect_overnight_data()
        
        if matches:
            output_file = args.output or f"data/overnight_{datetime.now().strftime('%Y%m%d')}.json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(matches, f, indent=2)
            
            end_time = time.time()
            duration = end_time - start_time
            logging.info(f"Saved {len(matches)} matches to {output_file}")
            logging.info(f"Collection completed in {duration:.2f} seconds ({duration/60:.2f} minutes)")
    else:
        monitor_subreddits(single_cycle=args.single)

if __name__ == '__main__':
    main()

