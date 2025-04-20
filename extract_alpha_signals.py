import json
from pathlib import Path
from datetime import datetime
import re

# Helper to parse GPT analysis field safely
def parse_gpt_analysis(raw):
    if isinstance(raw, dict):
        return raw
    elif isinstance(raw, str):
        try:
            raw = raw.strip().strip("`")
            match = re.search(r"\{.*\"confidence\"\s*:\s*\"(HIGH|MEDIUM|LOW)\".*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception as e:
            print("⚠️ Error parsing GPT analysis:", e)
    return {"analysis": str(raw), "confidence": "LOW"}

# Load latest GPT results
data_dir = Path("data")
gpt_files = sorted(data_dir.glob("gpt_insights_*.json"))
if not gpt_files:
    raise FileNotFoundError("❌ No gpt_insights_*.json file found in data/")

with open(gpt_files[-1]) as f:
    data = json.load(f)

print(f"📦 Loaded {len(data)} GPT-insight items from {gpt_files[-1].name}")

# Filter for high/medium confidence signals
from gpt_signal_matcher import generate_bet_recommendation  # Import the function if needed

filtered = []
for item in data:
    analysis = item.get("gpt_analysis", {})
    confidence = analysis.get("confidence", "LOW")
    if confidence in ["HIGH", "MEDIUM"]:
        recommendation = generate_bet_recommendation(
            analysis_text=analysis.get("analysis", ""),
            match_info=item.get("odds", {})
        )
        item["recommendation"] = recommendation
        filtered.append(item)

# Create a simpler summary version
simplified = []
for item in filtered:
    rec = item.get("recommendation")
    rationale = item.get("gpt_analysis", {}).get("rationale", "")
    risk = item.get("gpt_analysis", {}).get("risk", "")
    simplified.append({
        "match": f"{item['odds']['home_team']} vs {item['odds']['away_team']}",
        "recommendation": rec,
        "why": rationale,
        "risk": risk
    })

summary_path = data_dir / "alpha_bets_summary.json"
with open(summary_path, "w") as f:
    json.dump(simplified, f, indent=2)

# Save filtered results
alpha_out_path = data_dir / f"gpt_alpha_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(alpha_out_path, "w") as f:
    json.dump(filtered, f, indent=2)

print(f"✅ Saved summary to {summary_path}")
print(f"✅ Saved {len(filtered)} alpha signals to {alpha_out_path.resolve()}")
