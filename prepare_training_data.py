import json
from pathlib import Path
from team_aliases import TEAM_ALIASES
import re

def extract_teams(text):
    """Extract team names from text using team aliases"""
    teams = []
    text = text.lower()
    
    # Check for exact matches
    for team, aliases in TEAM_ALIASES.items():
        if team.lower() in text:
            teams.append(team)
            continue
            
        # Check aliases
        for alias in aliases:
            if alias.lower() in text:
                teams.append(team)
                break
    
    return teams

def prepare_team_identifier_data(matches_file, output_file):
    """Prepare data for team identifier model"""
    with open(matches_file, 'r', encoding='utf-8') as f:
        matches = json.load(f)
    
    training_data = []
    for match in matches:
        # Combine title and text
        text = f"{match.get('title', '')} {match.get('text', '')} {match.get('comment_text', '')}"
        
        # Extract teams
        teams = extract_teams(text)
        
        if teams:
            training_data.append({
                'text': text,
                'teams': teams
            })
    
    # Save training data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"Prepared {len(training_data)} samples for team identification")

def prepare_signal_reliability_data(matches_file, output_file):
    """Prepare data for signal reliability model"""
    with open(matches_file, 'r', encoding='utf-8') as f:
        matches = json.load(f)
    
    training_data = []
    for match in matches:
        # Extract features
        features = {
            'user_age_days': match.get('user_age_days', 0),
            'user_karma': match.get('user_karma', 0),
            'post_length': len(f"{match.get('title', '')} {match.get('text', '')}"),
            'has_links': bool(match.get('url')),
            'score': match.get('score', 0),
            'num_comments': match.get('num_comments', 0)
        }
        
        # Label based on score and comments
        label = 1 if features['score'] > 10 and features['num_comments'] > 5 else 0
        
        training_data.append({
            'features': features,
            'label': label
        })
    
    # Save training data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"Prepared {len(training_data)} samples for signal reliability")

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    Path("data/training").mkdir(parents=True, exist_ok=True)
    
    # Prepare data for both models
    prepare_team_identifier_data(
        "data/matches_20250420_093713.json",
        "data/training/team_identifier_data.json"
    )
    
    prepare_signal_reliability_data(
        "data/matches_20250420_093713.json",
        "data/training/signal_reliability_data.json"
    ) 