import json
from pathlib import Path
from team_aliases import TEAM_ALIASES
from datetime import datetime

def infer_team_from_text(text):
    text = text.lower()
    for team, aliases in TEAM_ALIASES.items():
        for alias in aliases:
            if alias in text:
                return team
    return None

# Load the latest Reddit file
reddit_dir = Path("data")
reddit_files = sorted(reddit_dir.glob("matches_*.json"))
latest_file = reddit_files[-1]
print(f"Enriching file: {latest_file.name}")

# Load data
with open(latest_file, "r") as f:
    posts = json.load(f)

# Enrich each post
for post in posts:
    full_text = (post.get("title", "") + " " + post.get("text", "")).lower()
    inferred_team = infer_team_from_text(full_text)
    if inferred_team:
        post["inferred_team"] = inferred_team

# Save to a new file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
enriched_file = reddit_dir / f"matches_enriched_{timestamp}.json"

with open(enriched_file, "w") as f:
    json.dump(posts, f, indent=2)

print(f"Enriched posts saved to: {enriched_file}")
