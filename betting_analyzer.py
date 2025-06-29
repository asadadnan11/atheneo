import json
from datetime import datetime
from config import BETTING_METRICS, SIGNAL_WEIGHTS, SOURCE_RELIABILITY, PERFORMANCE_METRICS, MARKET_DATA_CONFIG

class BettingAnalyzer:
    def __init__(self):
        self.performance_metrics = PERFORMANCE_METRICS.copy()
        self.market_data = {}
        self.signal_history = []
    
    def calculate_implied_probability(self, odds):
        # convert decimal odds to probability
        return 1 / odds
    
    def calculate_expected_value(self, probability, odds, stake=1):
        # basic EV calculation - profit * prob - loss * (1-prob)
        return (probability * (odds - 1) * stake) - ((1 - probability) * stake)
    
    def calculate_kelly_criterion(self, probability, odds):
        # kelly formula for optimal bet sizing
        if odds <= 1:
            return 0
        return (probability * odds - 1) / (odds - 1)
    
    def calculate_market_efficiency(self, odds_data):
        # check how efficient the market is based on total implied probability
        total_prob = sum(self.calculate_implied_probability(odds) for odds in odds_data.values())
        return 1 / total_prob if total_prob > 0 else 0
    
    def track_line_movements(self, odds_data, timestamp):
        # track when odds move significantly - could indicate sharp money
        if not self.market_data:
            self.market_data = {team: {'odds': odds, 'timestamp': timestamp} for team, odds in odds_data.items()}
            return []
        
        movements = []
        for team, current_odds in odds_data.items():
            if team in self.market_data:
                prev_odds = self.market_data[team]['odds']
                prev_time = self.market_data[team]['timestamp']
                time_diff = (timestamp - prev_time).total_seconds()
                
                if time_diff <= MARKET_DATA_CONFIG['time_window']:
                    odds_change = abs(current_odds - prev_odds)
                    if odds_change >= MARKET_DATA_CONFIG['min_odds_change']:
                        movements.append({
                            'team': team,
                            'previous_odds': prev_odds,
                            'current_odds': current_odds,
                            'change': odds_change,
                            'direction': 'up' if current_odds > prev_odds else 'down',
                            'time_diff': time_diff
                        })
        
        # update stored data
        self.market_data = {team: {'odds': odds, 'timestamp': timestamp} for team, odds in odds_data.items()}
        return movements
    
    def identify_sharp_money(self, line_movements, volume_data):
        # try to spot when sharp bettors are moving the line
        sharp_signals = []
        for movement in line_movements:
            if movement['change'] >= MARKET_DATA_CONFIG['min_odds_change'] * 2:  # big movement
                team_volume = volume_data.get(movement['team'], 0)
                if team_volume > 0:  # has betting activity
                    sharp_signals.append({
                        'team': movement['team'],
                        'movement': movement,
                        'volume': team_volume,
                        'confidence': min(1.0, movement['change'] * team_volume / 1000)  # rough confidence calc
                    })
        return sharp_signals
    
    def validate_signal(self, signal):
        # score how reliable a signal seems based on source and content
        reliability_score = SOURCE_RELIABILITY.get(signal.get('source_type', 'regular_users'), 0.5)
        
        # check signal content categories
        content_score = 0
        for category, weight in SIGNAL_WEIGHTS.items():
            if category in signal.get('categories', []):
                content_score += weight
        
        # combine scores
        final_score = reliability_score * content_score
        
        return {
            'reliability_score': reliability_score,
            'content_score': content_score,
            'final_score': final_score,
            'is_valid': final_score >= 0.5  # threshold for keeping signal
        }
    
    def track_performance(self, signal, outcome, stake, odds):
        # keep track of how our signals perform over time
        self.signal_history.append({
            'signal': signal,
            'outcome': outcome,
            'stake': stake,
            'odds': odds,
            'timestamp': datetime.now()
        })
        
        # basic performance stats
        total_signals = len(self.signal_history)
        winning_signals = sum(1 for s in self.signal_history if s['outcome'] == 'win')
        
        self.performance_metrics['win_rate'] = winning_signals / total_signals if total_signals > 0 else 0
        
        # profit/loss calculation
        profit_loss = sum(
            (s['odds'] - 1) * s['stake'] if s['outcome'] == 'win' else -s['stake']
            for s in self.signal_history
        )
        self.performance_metrics['profit_loss'] = profit_loss
        
        # roi calculation
        total_staked = sum(s['stake'] for s in self.signal_history)
        self.performance_metrics['roi'] = (profit_loss / total_staked) if total_staked > 0 else 0
        
        # track accuracy by signal type
        signal_types = set(s['signal'].get('type', 'unknown') for s in self.signal_history)
        for signal_type in signal_types:
            type_signals = [s for s in self.signal_history if s['signal'].get('type') == signal_type]
            type_wins = sum(1 for s in type_signals if s['outcome'] == 'win')
            self.performance_metrics['signal_accuracy'][signal_type] = type_wins / len(type_signals) if type_signals else 0
        
        return self.performance_metrics 