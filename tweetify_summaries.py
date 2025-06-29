from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from pathlib import Path
from datetime import datetime, timezone
import time
import re
from team_aliases import TEAM_ALIASES

class TweetifySummaries:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize OpenAI client with API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
    
    def is_match_active(self, commence_time):
        try:
            match_time = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
            current_time = datetime.now(timezone.utc)
            # only want matches in next 7 days
            return match_time > current_time and (match_time - current_time).days <= 7
        except Exception as e:
            print(f"Error checking match time: {e}")
            return False
    
    def generate_tweets(self, signals_input):
        """Generate tweets from signals.
        
        Args:
            signals_input: Either a file path (str) or a list of signal dictionaries
        """
        print("\nGenerating tweets from signals")
        
        # Handle different input types
        if isinstance(signals_input, str):
            try:
                with open(signals_input, 'r', encoding='utf-8') as f:
                    signals = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error reading signals file: {e}")
                return []
        elif isinstance(signals_input, list):
            signals = signals_input
        else:
            print(f"Invalid signals input type: {type(signals_input)}")
            return []
        
        print(f"Total signals loaded: {len(signals)}")
        
        tweets = []
        for i, signal in enumerate(signals, 1):
            print(f"\nProcessing signal {i}/{len(signals)}")
            
            # Get match info
            home_team = signal.get('odds', {}).get('home_team', '')
            away_team = signal.get('odds', {}).get('away_team', '')
            match_time = signal.get('odds', {}).get('commence_time', '')
            
            if not all([home_team, away_team, match_time]):
                print("Missing match details, skipping")
                continue
            
            if not self.is_match_active(match_time):
                print(f"Skipping expired match: {home_team} vs {away_team}")
                continue
            
            print(f"Processing match: {home_team} vs {away_team}")
            
            # Build prompt for GPT
            prompt = f"""
            Generate a concise betting tip tweet for this specific match:
            
            Match: {home_team} vs {away_team}
            Time: {match_time}
            
            Current Odds:
            - {home_team}: {signal.get('odds', {}).get('bookmakers', [{}])[0].get('markets', [{}])[0].get('outcomes', [{}])[0].get('price', 'N/A')}
            - {away_team}: {signal.get('odds', {}).get('bookmakers', [{}])[0].get('markets', [{}])[0].get('outcomes', [{}])[1].get('price', 'N/A')}
            - Draw: {signal.get('odds', {}).get('bookmakers', [{}])[0].get('markets', [{}])[0].get('outcomes', [{}])[2].get('price', 'N/A')}
            
            Analysis: {signal.get('gpt_analysis', {}).get('analysis', '')}
            
            Requirements:
            1. Must be under 280 characters
            2. Must ONLY mention {home_team} and {away_team}
            3. Focus ONLY on match-specific insights and current form
            4. DO NOT include any management changes, transfers, or news not directly related to this match
            5. Use emojis sparingly but effectively
            6. Include confidence level (Low/Medium/High) based ONLY on recent form and odds
            7. End with #BettingTip
            8. Include odds ONLY if they are competitive and relevant
            9. Stick to factual, verifiable information about these two teams
            10. DO NOT make speculative claims about team changes or future events
            """
            
            try:
                print("Generating tweet with GPT...")
                resp = self.client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[{"role": "system", "content": "You are a sports betting analyst."},
                            {"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=150
                )
                
                tweet = resp.choices[0].message.content.strip()
                print(f"Generated tweet: {tweet}")
                
                # Basic validation
                if len(tweet) > 280:
                    print(f"Tweet too long ({len(tweet)} chars), skipping")
                    continue
                
                if not all(team.lower() in tweet.lower() for team in [home_team, away_team]):
                    print("Tweet missing team names, skipping")
                    continue
                
                tweets.append({
                    'tweet': tweet,
                    'home_team': home_team,
                    'away_team': away_team,
                    'match_time': match_time,
                    'generated_at': datetime.now().isoformat()
                })
                print("Tweet validated and added")
                
                # Rate limit protection
                time.sleep(1.2)
                
            except Exception as e:
                print(f"Error generating tweet: {e}")
                continue
        
        # Save results
        if tweets:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            tweets_file = f'data/alpha_bets_tweets_{timestamp}.json'
            tweets_text_file = f'data/alpha_bets_tweets_{timestamp}.txt'
            
            os.makedirs('data', exist_ok=True)
            
            with open(tweets_file, 'w', encoding='utf-8') as f:
                json.dump(tweets, f, indent=2)
            
            with open(tweets_text_file, 'w', encoding='utf-8') as f:
                for tweet_data in tweets:
                    f.write(tweet_data['tweet'] + '\n\n')
            
            print(f"\nSaved {len(tweets)} tweets to:")
            print(f"JSON: {tweets_file}")
            print(f"Text: {tweets_text_file}")
        else:
            print("\nNo valid tweets were generated")
        
        return tweets

# Run this if called directly
if __name__ == "__main__":
    # Find the most recent GPT insights file
    data_dir = Path("data")
    gpt_files = sorted(data_dir.glob("gpt_insights_*.json"))
    
    if not gpt_files:
        print("No GPT insights files found in data directory")
        exit(1)
        
    latest_file = gpt_files[-1]
    print(f"Using latest insights file: {latest_file}")
    
    tweetify = TweetifySummaries()
    tweetify.generate_tweets(latest_file)
