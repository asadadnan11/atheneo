# Betting Configuration
BETTING_METRICS = {
    'implied_probability': True,
    'expected_value': True,
    'kelly_criterion': True,
    'market_efficiency': True
}

# Signal Weights
SIGNAL_WEIGHTS = {
    'injury': 0.3,
    'lineup': 0.2,
    'form': 0.15,
    'weather': 0.1,
    'betting_market': 0.25
}

# Source Reliability
SOURCE_RELIABILITY = {
    'verified_users': 1.0,
    'moderators': 0.9,
    'regular_users': 0.7,
    'new_users': 0.5
}

# Performance Tracking
PERFORMANCE_METRICS = {
    'win_rate': 0.0,
    'profit_loss': 0.0,
    'roi': 0.0,
    'signal_accuracy': {}
}

# Market Data Processing
MARKET_DATA_CONFIG = {
    'track_line_movements': True,
    'calculate_market_efficiency': True,
    'identify_sharp_money': True,
    'min_odds_change': 0.1,  # Minimum odds change to track
    'time_window': 3600,     # 1 hour window for tracking changes
} 