# WhisperBet: AI-Powered Sports Betting Insights

## Overview
WhisperBet is a comprehensive system that combines Reddit sentiment analysis with betting market data to identify potential betting opportunities. It processes social media signals, matches them with betting odds, and generates actionable insights using advanced betting metrics. The system includes a Streamlit app for visualizing insights and sentiment analysis.

## Features

### 1. Real-time Betting Insights Dashboard
- **Overview Metrics**: Total signals, active matches, and confidence scores
- **Sentiment Distribution**: Visual breakdown of betting sentiment (strongly positive to strongly negative)
- **Team Analysis**: Track trending teams and market sentiment
- **Latest Tips**: Real-time betting recommendations with match details

### 2. Advanced Sentiment Analysis
- **Multi-level Classification**: 
  - Strongly Positive: High confidence value bets
  - Positive: Good betting opportunities
  - Neutral: Balanced or unclear signals
  - Negative: Poor value or high risk
  - Strongly Negative: Strong avoid signals
- **Confidence Scoring**: 
  - Automated confidence assessment (0-1 scale)
  - Based on signal strength and market consensus
  - Weighted by source reliability

### 3. Team Analysis Features
- **Mention Tracking**: Monitor team discussion frequency
- **Pattern Recognition**: Identify team aliases and nicknames
- **Context Analysis**: Understand team references in various formats

### 4. Data Collection & Processing
- **Reddit Integration**: 
  - Monitors key subreddits for betting signals
  - Tracks team news, injuries, and lineups
  - Filters relevant betting discussions
- **Signal Processing**:
  - Team identification using pattern matching
  - Odds matching with current markets
  - Signal validation and reliability scoring
- **Betting Analysis**:
  - Implied probability calculations
  - Expected value analysis
  - Kelly Criterion for optimal staking

## Getting Started

### Prerequisites
- Python 3.8+
- Reddit API credentials (see below for setup)
- OpenAI API key
- Git LFS (for model files)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/whisperbet.git
   cd whisperbet
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables in `.env`:
   ```
   REDDIT_CLIENT_ID=your_client_id
   REDDIT_CLIENT_SECRET=your_client_secret
   OPENAI_API_KEY=your_api_key
   ```

4. Initialize the system:
   ```bash
   python reddit_harvester.py  # Start data collection
   python gpt_signal_matcher.py  # Process signals
   ```

5. Launch the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

### Reddit API Setup
1. Go to https://www.reddit.com/prefs/apps
2. Click "Create App" or "Create Another App"
3. Select "script"
4. Fill in the required information
5. Copy the client ID and client secret to your `.env` file

## Using the Dashboard

### Overview Tab
- View key metrics and sentiment distribution
- Monitor active signals and confidence scores
- Track overall market sentiment

### Analysis Tab
- Explore team-specific analysis
- View trending teams and market movement
- Analyze historical patterns

### Tweets Tab
- See latest betting recommendations
- View detailed match information
- Track confidence levels and reasoning

## Configuration
- `config.py`: Adjust betting metrics and thresholds
- `team_aliases.py`: Customize team identification patterns
- `.env`: Set up API credentials and environment variables
- `.streamlit/config.toml`: Customize Streamlit app appearance

## Troubleshooting

### Common Issues
1. **API Rate Limits**:
   - Implement proper delays between Reddit API calls
   - Use caching for frequent requests

2. **Model Loading Errors**:
   - Ensure Git LFS properly downloaded model files
   - Check model file paths in config

3. **Sentiment Analysis Issues**:
   - Verify confidence thresholds in config
   - Check pattern matching rules

4. **Dashboard Performance**:
   - Enable caching for heavy computations
   - Optimize data loading patterns

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

## Support
For issues and feature requests, please use the GitHub issue tracker. 