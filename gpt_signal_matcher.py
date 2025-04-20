import json
from pathlib import Path
from datetime import datetime
from team_aliases import TEAM_ALIASES
from dotenv import load_dotenv
import os
import re
import openai
from betting_analyzer import BettingAnalyzer

class GPTSignalMatcher:
    def __init__(self):
        # Force creation of data directory if it doesn't exist
        Path("data").mkdir(parents=True, exist_ok=True)
        
        # Load environment variables
        load_dotenv()
        
        # Verify API key is loaded
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        print("✅ OpenAI API key loaded successfully")
        
        # Set the API key for the older version of the package
        openai.api_key = api_key
        
        # Initialize betting analyzer
        self.betting_analyzer = BettingAnalyzer()
    
    def infer_team_from_text(self, text):
        """Infer team from text, checking both title and comment text"""
        if not text:
            return None
            
        text = text.lower()
        
        # First check subreddit name if available
        subreddit = text.split('subreddit": "')[-1].split('"')[0] if 'subreddit": "' in text else ""
        if subreddit:
            # Map common subreddit names to teams
            subreddit_map = {
                'reddevils': 'manchester united',
                'gunners': 'arsenal',
                'coys': 'tottenham',
                'mcfc': 'manchester city',
                'chelseafc': 'chelsea',
                'liverpoolfc': 'liverpool'
            }
            if subreddit.lower() in subreddit_map:
                return subreddit_map[subreddit.lower()]
        
        # Try exact matches
        for team, aliases in TEAM_ALIASES.items():
            # Check team name itself
            if team.lower() in text:
                return team
                
            # Check aliases (exact matches)
            for alias in aliases:
                if alias.lower() in text:
                    return team
        
        # If no exact matches, try partial matches but be more strict
        for team, aliases in TEAM_ALIASES.items():
            team_parts = team.lower().split()
            # Only match if all parts of the team name are present
            if all(part in text for part in team_parts):
                return team
                
            # Check aliases but require longer matches
            for alias in aliases:
                alias_parts = alias.lower().split()
                # Only match if the alias is at least 5 chars and all parts are present
                if len(alias) >= 5 and all(part in text for part in alias_parts):
                    return team
        
        return None
    
    def ask_gpt(self, post, odds):
        # Get the odds values
        market = odds['bookmakers'][0]['markets'][0]
        home_odds = next(o['price'] for o in market['outcomes'] if o['name'] == odds['home_team'])
        away_odds = next(o['price'] for o in market['outcomes'] if o['name'] == odds['away_team'])
        
        # Calculate betting metrics
        home_prob = self.betting_analyzer.calculate_implied_probability(home_odds)
        away_prob = self.betting_analyzer.calculate_implied_probability(away_odds)
        market_efficiency = self.betting_analyzer.calculate_market_efficiency({
            odds['home_team']: home_odds,
            odds['away_team']: away_odds
        })
        
        prompt = f"""
You are a sharp sports betting analyst who hunts for edge. Return ONLY a valid JSON object with no additional text.

Given the Reddit post and betting odds below, analyze and return a JSON object with these exact keys:

{{
  "analysis": "What the post reveals (injury, morale, surprise lineup, etc.)",
  "confidence": "HIGH" or "MEDIUM" or "LOW",
  "action": "A betting suggestion like 'Bet on X', 'Avoid market', or 'Wait for news'",
  "rationale": "Why this bet makes sense given the context and odds",
  "risk": "What could go wrong — e.g., public info, inaccurate rumor, team rotation",
  "betting_metrics": {{
    "implied_probability": {{
      "home": {home_prob:.3f},
      "away": {away_prob:.3f}
    }},
    "market_efficiency": {market_efficiency:.3f},
    "expected_value": {{
      "home": {self.betting_analyzer.calculate_expected_value(home_prob, home_odds):.3f},
      "away": {self.betting_analyzer.calculate_expected_value(away_prob, away_odds):.3f}
    }},
    "kelly_criterion": {{
      "home": {self.betting_analyzer.calculate_kelly_criterion(home_prob, home_odds):.3f},
      "away": {self.betting_analyzer.calculate_kelly_criterion(away_prob, away_odds):.3f}
    }}
  }}
}}

Reddit Post:
Title: {post.get("title", "")}
Body: {post.get("text", "")}

Match:
{odds['home_team']} vs {odds['away_team']}
Start time: {odds['commence_time']}

Odds:
{odds['home_team']}: {home_odds}
{odds['away_team']}: {away_odds}

IMPORTANT:
- Return ONLY the JSON object
- Do not include any markdown formatting
- Do not include any explanations
- Ensure all values are properly escaped
- Do not include any text before or after the JSON

Be aggressive. Look for injuries, fatigue, revenge narratives, surprise lineups, or morale issues not priced into the market.
"""

        print(f"\n⏳ Asking GPT for: {post.get('title', '')[:60]}...")

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a sharp sports betting analyst. Return only valid JSON with no additional text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3  # Lower temperature for more consistent JSON
            )

            raw_reply = response.choices[0].message.content.strip()
            
            # Clean up the response
            raw_reply = re.sub(r"```(?:json)?", "", raw_reply)
            raw_reply = raw_reply.replace("```", "").strip()
            
            # Try to parse the JSON
            try:
                result = json.loads(raw_reply)
                # Validate required keys
                required_keys = ["analysis", "confidence", "action", "rationale", "risk", "betting_metrics"]
                if all(key in result for key in required_keys):
                    return result
                else:
                    print(f"❌ Missing required keys in GPT response")
                    return self._create_default_response()
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {str(e)}")
                return self._create_default_response()
                
        except Exception as e:
            print(f"❌ Error in GPT API call: {str(e)}")
            return self._create_default_response()
    
    def _create_default_response(self):
        """Create a default response when GPT fails"""
        return {
            "analysis": "Failed to analyze post",
            "confidence": "LOW",
            "action": "No actionable insight",
            "rationale": "Technical error in analysis",
            "risk": "Unable to assess risks",
            "betting_metrics": {
                "implied_probability": {"home": 0.0, "away": 0.0},
                "market_efficiency": 0.0,
                "expected_value": {"home": 0.0, "away": 0.0},
                "kelly_criterion": {"home": 0.0, "away": 0.0}
            }
        }
    
    def match_team_to_post(self, post, odds_data):
        # Get the team from either title or comment
        team = post.get("inferred_team")
        if not team:
            # Try to get team from parent title and comment text
            full_text = post.get("parent_title", "") + " " + post.get("comment_text", "")
            if not full_text.strip():
                full_text = post.get("title", "") + " " + post.get("text", "")
            
            # Add subreddit info to the text
            full_text += " subreddit: " + post.get("subreddit", "")
            
            team = self.infer_team_from_text(full_text)
            if not team:
                print(f"⛔ No team found in: {post.get('parent_title', post.get('title', ''))[:60]}")
                return None

        print(f"🔍 Looking for matches for team: {team}")
        for match in odds_data:
            home = match['home_team'].lower()
            away = match['away_team'].lower()
            team_lower = team.lower()
            
            # Check for exact match first
            if team_lower in [home, away]:
                print(f"✅ Exact match found: {team} in {home} vs {away}")
                return self._create_simplified_match(match)
            
            # Then check for strict partial matches
            team_parts = team_lower.split()
            if len(team_parts) > 1:  # Only do partial matches for multi-word teams
                if (all(part in home for part in team_parts) or 
                    all(part in away for part in team_parts)):
                    print(f"✅ Partial match found: {team} in {home} vs {away}")
                    return self._create_simplified_match(match)

        print(f"❌ No match for post with team '{team}'")
        return None
        
    def _create_simplified_match(self, match):
        """Create a simplified match structure from the odds data"""
        bookmaker = match['bookmakers'][0]
        market = bookmaker['markets'][0]  # Get h2h market
        
        return {
            'home_team': match['home_team'],
            'away_team': match['away_team'],
            'commence_time': match['commence_time'],
            'bookmakers': [
                {
                    'key': bookmaker['key'],
                    'markets': [
                        {
                            'key': market['key'],
                            'outcomes': market['outcomes']
                        }
                    ]
                }
            ]
        }
    
    def process_matches(self, matches):
        print(f"\n📊 Total matches to process: {len(matches)}")
        
        # Load latest Odds file
        odds_dir = Path("data")
        odds_files = sorted(odds_dir.glob("odds_*.json"))
        if not odds_files:
            raise FileNotFoundError("No odds files found")
        
        # Get the most recent odds file
        latest_odds_file = odds_files[-1]
        print(f"📊 Loading odds from: {latest_odds_file}")
        
        with open(latest_odds_file, 'r', encoding='utf-8') as f:
            odds_data = json.load(f)
        print(f"📊 Total odds data loaded: {len(odds_data)} matches")
        
        print(f"\n🔍 Processing {len(matches)} matches...")
        
        # Tag posts with inferred team
        for post in matches:
            full_text = (post.get("title", "") + " " + post.get("text", ""))
            post["inferred_team"] = self.infer_team_from_text(full_text)
            if post["inferred_team"]:
                print(f"✅ Found team '{post['inferred_team']}' in: {post.get('title', '')[:60]}")
        
        # Run GPT analysis
        results = []
        for post in matches:
            if not post.get("inferred_team"):
                continue
                
            odds = self.match_team_to_post(post, odds_data)
            if odds:
                print(f"📊 Processing match: {odds['home_team']} vs {odds['away_team']}")
                
                # Validate signal
                signal_validation = self.betting_analyzer.validate_signal({
                    'source_type': 'regular_users',  # Default, can be improved
                    'categories': self._extract_categories(post),
                    'content': post
                })
                
                if signal_validation['is_valid']:
                    analysis = self.ask_gpt(post, odds)
                    results.append({
                        "post": post,
                        "odds": odds,
                        "gpt_analysis": analysis,
                        "signal_validation": signal_validation
                    })
                    print(f"✅ Added analysis for: {odds['home_team']} vs {odds['away_team']}")
        
        print(f"\n✅ GPT analysis complete. Processed {len(results)} matches")
        return results
    
    def _extract_categories(self, post):
        """Extract relevant categories from post content"""
        categories = []
        text = (post.get("title", "") + " " + post.get("text", "")).lower()
        
        if any(keyword in text for keyword in ['injury', 'injured', 'fitness']):
            categories.append('injury')
        if any(keyword in text for keyword in ['lineup', 'starting xi', 'team news']):
            categories.append('lineup')
        if any(keyword in text for keyword in ['form', 'momentum', 'confidence']):
            categories.append('form')
        if any(keyword in text for keyword in ['weather', 'pitch', 'condition']):
            categories.append('weather')
        if any(keyword in text for keyword in ['odds', 'value', 'edge', 'bet']):
            categories.append('betting_market')
            
        return categories

# For backward compatibility
if __name__ == "__main__":
    matcher = GPTSignalMatcher()
    
    # Load the latest matches file
    matches_dir = Path("data")
    matches_files = sorted(matches_dir.glob("matches_*.json"))
    if not matches_files:
        raise FileNotFoundError("No matches files found")
    
    # Get the most recent matches file
    latest_matches_file = matches_files[-1]
    print(f"📊 Loading matches from: {latest_matches_file}")
    
    with open(latest_matches_file, 'r', encoding='utf-8') as f:
        matches = json.load(f)
    
    results = matcher.process_matches(matches)
    print(f"\n✅ GPT analysis complete. Processed {len(results)} matches")




