import requests
import json
import os
from datetime import datetime, timedelta

API_KEY = '72f26f919211dd0a2fd79a6f688dc256'  # replace this

# Define all leagues we want to track
LEAGUES = {
    'soccer_epl': 'English Premier League',
    'soccer_austria_bundesliga': 'Austrian Bundesliga',
    'soccer_serbia_superliga': 'Serbian SuperLiga',
    'soccer_spain_la_liga': 'La Liga',
    'soccer_germany_bundesliga': 'Bundesliga',
    'soccer_italy_serie_a': 'Serie A',
    'soccer_france_ligue_one': 'Ligue 1',
    'soccer_turkey_super_league': 'Turkish Süper Lig',
    'soccer_portugal_primeira_liga': 'Portuguese Primeira Liga'
}

REGION = 'uk'  # options: 'uk', 'us', 'eu'
MARKET = 'h2h'  # head-to-head (moneyline)

# Ensure directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('data/history', exist_ok=True)

def cleanup_old_files():
    """Clean up old odds files, keeping only the last 7 days"""
    try:
        for file in os.listdir('data/history'):
            if file.startswith('odds_'):
                file_path = os.path.join('data/history', file)
                file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                if (datetime.now() - file_time).days > 7:
                    os.remove(file_path)
    except Exception as e:
        print(f"Error cleaning up old files: {str(e)}")

def fetch_odds_for_league(league_id):
    """Fetch odds for a specific league"""
    url = f"https://api.the-odds-api.com/v4/sports/{league_id}/odds"
    params = {
        'apiKey': API_KEY,
        'regions': REGION,
        'markets': MARKET,
        'oddsFormat': 'decimal'
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Error fetching odds for {league_id}: {response.status_code}, {response.text}")
        return []
    
    return response.json()

# Fetch odds for all leagues
all_odds = []
for league_id, league_name in LEAGUES.items():
    print(f"Fetching odds for {league_name}...")
    odds = fetch_odds_for_league(league_id)
    all_odds.extend(odds)
    print(f"Found {len(odds)} matches")

# Save current odds
current_file = "data/current_odds.json"
with open(current_file, 'w') as f:
    json.dump(all_odds, f, indent=2)

# Save historical copy
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
history_file = f"data/history/odds_{timestamp}.json"
with open(history_file, 'w') as f:
    json.dump(all_odds, f, indent=2)

# Clean up old files
cleanup_old_files()

print(f"✅ Saved {len(all_odds)} total match odds to:")
print(f"📄 Current: {current_file}")
print(f"📄 History: {history_file}")

