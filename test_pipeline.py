import os
import json
import time
import logging
from datetime import datetime
from gpt_signal_matcher import GPTSignalMatcher
from tweetify_summaries import TweetifySummaries

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Ensure data directory exists
os.makedirs('data', exist_ok=True)
os.makedirs('data/test_pipeline', exist_ok=True)

def wait_for_matches_file(timeout=300, check_interval=10, max_retries=3):
    """Wait for matches file to be created and contain data"""
    matches_file = 'data/current_matches.json'
    start_time = time.time()
    retries = 0
    
    while time.time() - start_time < timeout:
        if os.path.exists(matches_file):
            try:
                with open(matches_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:  # Empty file
                        logging.info("Matches file exists but is empty")
                        if retries < max_retries:
                            retries += 1
                            time.sleep(check_interval)
                            continue
                        else:
                            raise ValueError("Matches file is empty after max retries")
                    
                    try:
                        matches = json.loads(content)
                        if matches:
                            if not isinstance(matches, list):
                                matches = list(matches)
                            logging.info(f"Found {len(matches)} matches in file")
                            return matches
                        else:
                            logging.info("Matches file contains empty list")
                    except json.JSONDecodeError as je:
                        logging.warning(f"Invalid JSON in matches file: {str(je)}")
                        if retries < max_retries:
                            retries += 1
                            time.sleep(check_interval)
                            continue
                        else:
                            raise
            except Exception as e:
                logging.error(f"Error reading matches file: {str(e)}")
                if retries < max_retries:
                    retries += 1
                    time.sleep(check_interval)
                    continue
                else:
                    raise
        
        logging.info(f"Waiting for matches file... (timeout in {int(timeout - (time.time() - start_time))}s)")
        time.sleep(check_interval)
    
    raise TimeoutError("No matches found in the file. Make sure the Reddit harvester is running.")

def main():
    try:
        # Wait for matches file
        matches = wait_for_matches_file()
        
        # Initialize components
        signal_matcher = GPTSignalMatcher()
        tweetify = TweetifySummaries()
        
        # Process matches
        signals = signal_matcher.process_matches(matches)
        
        # Save signals
        signals_file = 'data/test_pipeline/signals.json'
        with open(signals_file, 'w', encoding='utf-8') as f:
            json.dump(signals, f, indent=2)
        logging.info(f"Saved {len(signals)} signals to {signals_file}")
        
        # Generate tweets
        tweets = tweetify.generate_tweets(signals)
        
        # Save tweets
        tweets_file = 'data/test_pipeline/tweets.json'
        with open(tweets_file, 'w', encoding='utf-8') as f:
            json.dump(tweets, f, indent=2)
        logging.info(f"Saved {len(tweets)} tweets to {tweets_file}")
        
        logging.info("Pipeline completed successfully")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 