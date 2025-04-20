# WhisperBet: AI-Powered Sports Betting Insights

## Overview
WhisperBet is a comprehensive system that combines Reddit sentiment analysis with betting market data to identify potential betting opportunities. It processes social media signals, matches them with betting odds, and generates actionable insights using advanced betting metrics. The system includes a Streamlit app for visualizing insights and sentiment analysis.

## System Architecture

### 1. Data Collection (Reddit Harvester)
- Monitors key subreddits for relevant information
- Tracks team news, injuries, lineups, and betting discussions
- Uses keyword-based filtering for signal detection
- Stores historical data for analysis

### 2. Signal Processing (GPTSignalMatcher)
- Team identification using pattern matching and aliases
- Odds matching with current betting markets
- Signal validation based on source reliability
- Integration with GPT for contextual analysis

### 3. Betting Analysis (BettingAnalyzer)
- Implied probability calculations
- Expected value analysis
- Kelly Criterion for optimal staking
- Market efficiency scoring
- Line movement tracking
- Performance metrics tracking

### 4. Visualization (Streamlit App)
- **Sentiment Analysis**: Visualizes betting sentiment distribution and confidence scores
- **Team Analysis**: Tracks team mentions and trends
- **Latest Betting Tips**: Displays recent betting recommendations with match details
- **Overview Metrics**: Provides real-time updates on signals and confidence

## Future ML Improvements

### Phase 1: Signal Quality Enhancement
1. **Team Identification Model**
   - Train on historical team references
   - Improve nickname/abbreviation recognition
   - Better contextual understanding
   - Implementation priority: High

2. **Signal Reliability Model**
   - User history analysis
   - Post pattern recognition
   - Historical accuracy tracking
   - Implementation priority: High

### Phase 2: Market Analysis
3. **Market Movement Predictor**
   - Historical pattern analysis
   - Volume impact assessment
   - News reaction modeling
   - Implementation priority: Medium

4. **Advanced Sentiment Analysis**
   - Context-aware sentiment scoring
   - Sarcasm/irony detection
   - Confidence level assessment
   - Implementation priority: Medium

### Phase 3: Performance Optimization
5. **Bet Sizing Model**
   - Dynamic Kelly Criterion
   - Bankroll management
   - Risk-adjusted sizing
   - Implementation priority: Medium

6. **Strategy Adaptation**
   - Market condition analysis
   - Pattern recognition
   - Performance optimization
   - Implementation priority: Low

## Getting Started

### Prerequisites
- Python 3.8+
- Reddit API credentials
- OpenAI API key
- Environment variables setup

### Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables in `.env`
4. Run the Reddit harvester: `python reddit_harvester.py`
5. Process matches: `python gpt_signal_matcher.py`
6. Launch the Streamlit app: `streamlit run streamlit_app.py`

## Configuration
- Edit `config.py` for betting metrics settings
- Adjust `team_aliases.py` for team identification
- Modify signal weights in configuration

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
MIT License

## Roadmap
1. Implement ML-based team identification
2. Add signal reliability scoring
3. Develop market movement prediction
4. Enhance sentiment analysis
5. Optimize bet sizing
6. Add strategy adaptation 