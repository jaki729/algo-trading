# Complete Integrated Live Trading System
import asyncio
import aiohttp
import concurrent.futures
from datetime import datetime, timedelta
import logging
import threading
import time
import signal
import sys
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import queue

# Import all previous section
# from rest_api_client import AlpacaRESTClient, APIResponse
# from order_management_system import OrderManager, OrderRequest, Position
# from market_data_validation import MarketDataValidator, AlpacaWebSocketClient, MarketDataPoint
# from performance_risk_reporting import DatabaseManager, PerformanceAnalyzer, RiskDashboard, TradeRecord
# from rest_api_client import AlpacaRESTClient, APIResponse
# from order_management_system import OrderManager, OrderRequest, Position
# from market_data_validation import MarketDataValidator, AlpacaWebSocketClient, MarketDataPoint
# from performance_risk_reporting import DatabaseManager, PerformanceAnalyzer, RiskDashboard, TradeRecord
logger = logging.getLogger(__name__)

@dataclass
class StrategySignal:
    """Trading strategy signal"""
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: float  # 0.0 to 1.0
    strategy_name: str
    timestamp: datetime
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    confidence: float = 0.0
    additional_data: Dict = None

class StrategyEngine:
    """Semiconductor trading strategy engine implementing strategies"""
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.signals: Dict[str, StrategySignal] = {}
        
        # Trend Following Parameters (from Part 1)
        self.ema_periods = [5, 8, 13, 21, 34, 55]
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.adx_period = 14
        
        # Mean Reversion Parameters (from Part 1)
        self.z_score_windows = [10, 20, 50]
        self.rsi_period = 14
        self.bollinger_period = 20
        self.bollinger_std = 2
        self.ou_window = 50  # Ornstein-Uhlenbeck window
        
        # Technical indicators storage
        self.technical_indicators = defaultdict(dict)
    
    def update_market_data(self, data_point: MarketDataPoint):
        """Update strategy with new market data"""
        symbol = data_point.symbol
        
        self.price_history[symbol].append(data_point.price)
        self.volume_history[symbol].append(data_point.volume)
        
        # Calculate technical indicators
        self._calculate_all_indicators(symbol)
        
        # Generate signals if we have enough data
        if len(self.price_history[symbol]) >= max(self.ema_periods):
            trend_signal = self._calculate_trend_signal(symbol)
            mean_rev_signal = self._calculate_mean_reversion_signal(symbol)
            
            # Combine signals using AL_JAKIUR_RAHMAN methodology
            combined_signal = self._combine_signals(trend_signal, mean_rev_signal)
            
            self.signals[symbol] = combined_signal
    
    def _calculate_all_indicators(self, symbol: str):
        """Calculate all technical indicators for a symbol"""
        prices = list(self.price_history[symbol])
        volumes = list(self.volume_history[symbol])
        
        if len(prices) < 20:
            return
        
        indicators = {}
        
        # EMAs (from Part 1 Trend Following)
        for period in self.ema_periods:
            if len(prices) >= period:
                indicators[f'EMA_{period}'] = self._calculate_ema(prices, period)
        
        # MACD (from Part 1)
        if len(prices) >= self.macd_slow:
            macd_line, macd_signal_line, macd_histogram = self._calculate_macd(prices)
            indicators['MACD'] = macd_line
            indicators['MACD_signal'] = macd_signal_line
            indicators['MACD_histogram'] = macd_histogram
        
        # ADX (from Part 1)
        if len(prices) >= self.adx_period + 10:
            indicators['ADX'] = self._calculate_adx(prices)
        
        # RSI (from Part 1 Mean Reversion)
        if len(prices) >= self.rsi_period + 1:
            indicators['RSI'] = self._calculate_rsi(prices, self.rsi_period)
        
        # Z-Scores (from Part 1 Mean Reversion)
        for window in self.z_score_windows:
            if len(prices) >= window:
                indicators[f'Z_Score_{window}'] = self._calculate_z_score(prices, window)
        
        # Bollinger Bands (from Part 1)
        if len(prices) >= self.bollinger_period:
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(prices)
            indicators['BB_upper'] = bb_upper
            indicators['BB_middle'] = bb_middle
            indicators['BB_lower'] = bb_lower
            indicators['BB_position'] = (prices[-1] - bb_lower) / (bb_upper - bb_lower)
        
        # Ornstein-Uhlenbeck Process (from Part 1 Mean Reversion)
        if len(prices) >= self.ou_window:
            ou_theta, ou_mu, ou_sigma = self._calculate_ou_process(prices)
            indicators['OU_theta'] = ou_theta
            indicators['OU_mu'] = ou_mu
            indicators['OU_sigma'] = ou_sigma
        
        self.technical_indicators[symbol] = indicators
    
    def _calculate_trend_signal(self, symbol: str) -> StrategySignal:
        """
        Implements EMA, MACD, ADX, and Volume Confirmation
        """
        indicators = self.technical_indicators[symbol]
        prices = list(self.price_history[symbol])
        current_price = prices[-1]
        
        # Initialize signal components
        ema_signal = 0.0
        macd_signal = 0.0
        adx_signal = 0.0
        strength_signal = 0.0
        
        # 1. EMA Crossover Signals (from Part 1)
        if 'EMA_8' in indicators and 'EMA_21' in indicators and 'EMA_55' in indicators:
            ema_8 = indicators['EMA_8']
            ema_21 = indicators['EMA_21']
            ema_55 = indicators['EMA_55']
            
            # EMA ratio method (from Part 1)
            ema_ratio_short = ema_8 / ema_21 if ema_21 > 0 else 1.0
            ema_ratio_medium = ema_21 / ema_55 if ema_55 > 0 else 1.0
            
            # Generate EMA signals
            if ema_ratio_short > 1.005 and ema_ratio_medium > 1.002:
                ema_signal = 1.0  # Strong uptrend
            elif ema_ratio_short > 1.001:
                ema_signal = 0.5  # Weak uptrend
            elif ema_ratio_short < 0.995 and ema_ratio_medium < 0.998:
                ema_signal = -1.0  # Strong downtrend
            elif ema_ratio_short < 0.999:
                ema_signal = -0.5  # Weak downtrend
        
        # 2. MACD Signals (from Part 1)
        if 'MACD' in indicators and 'MACD_signal' in indicators:
            macd_line = indicators['MACD']
            macd_signal_line = indicators['MACD_signal']
            
            if macd_line > macd_signal_line and macd_line > 0:
                macd_signal = 1.0  # Buy signal
            elif macd_line < macd_signal_line and macd_line < 0:
                macd_signal = -1.0  # Sell signal
            elif macd_line > macd_signal_line:
                macd_signal = 0.5  # Weak buy
            elif macd_line < macd_signal_line:
                macd_signal = -0.5  # Weak sell
        
        # 3. ADX Trend Strength (from Part 1)
        if 'ADX' in indicators:
            adx = indicators['ADX']
            if adx > 25:
                strength_signal = 1.0  # Strong trend
            elif adx > 20:
                strength_signal = 0.7  # Moderate trend
            else:
                strength_signal = 0.3  # Weak trend
        
        # 4. Volume Confirmation (from Part 1)
        volume_signal = 1.0
        if len(self.volume_history[symbol]) >= 20:
            volumes = list(self.volume_history[symbol])
            avg_volume = np.mean(volumes[-20:])
            current_volume = volumes[-1]
            if current_volume > avg_volume * 1.2:
                volume_signal = 1.2  # High volume confirmation
            elif current_volume < avg_volume * 0.8:
                volume_signal = 0.8  # Low volume warning
        
        # Combine trend signals using Part 1 methodology
        combined_strength = (
            ema_signal * 0.4 +          # EMA crossovers (40%)
            macd_signal * 0.3 +         # MACD (30%)
            (ema_signal * strength_signal) * 0.2 +  # Trend strength (20%)
            (ema_signal * 0.1)          # Additional EMA weight (10%)
        ) * volume_signal
        
        # Determine signal type and confidence
        if combined_strength > 0.6:
            signal_type = 'buy'
            confidence = min(combined_strength, 1.0)
        elif combined_strength < -0.6:
            signal_type = 'sell'
            confidence = min(abs(combined_strength), 1.0)
        else:
            signal_type = 'hold'
            confidence = 0.0
        
        return StrategySignal(
            symbol=symbol,
            signal_type=signal_type,
            strength=abs(combined_strength),
            strategy_name="TrendFollowing",
            timestamp=datetime.now(),
            confidence=confidence,
            additional_data={
                'ema_signal': ema_signal,
                'macd_signal': macd_signal,
                'adx_strength': strength_signal,
                'volume_confirmation': volume_signal
            }
        )
    
    def _calculate_mean_reversion_signal(self, symbol: str) -> StrategySignal:
        """
        Implements Z-Score, RSI, Bollinger Bands, and Ornstein-Uhlenbeck process
        """
        indicators = self.technical_indicators[symbol]
        prices = list(self.price_history[symbol])
        current_price = prices[-1]
        
        # Initialize signal components
        z_score_signal = 0.0
        rsi_signal = 0.0
        bb_signal = 0.0
        ou_signal = 0.0
        
        # 1. Z-Score Signals (from Part 1)
        z_signals = []
        for window in self.z_score_windows:
            if f'Z_Score_{window}' in indicators:
                z_score = indicators[f'Z_Score_{window}']
                if z_score <= -2.0:
                    z_signals.append(1.0)  # Oversold
                elif z_score >= 2.0:
                    z_signals.append(-1.0)  # Overbought
                elif z_score <= -1.5:
                    z_signals.append(0.5)   # Weakly oversold
                elif z_score >= 1.5:
                    z_signals.append(-0.5)  # Weakly overbought
                else:
                    z_signals.append(0.0)
        
        if z_signals:
            z_score_signal = np.mean(z_signals)
        
        # 2. RSI Signals (from Part 1)
        if 'RSI' in indicators:
            rsi = indicators['RSI']
            if rsi <= 30:
                rsi_signal = 1.0    # Oversold - buy
            elif rsi >= 70:
                rsi_signal = -1.0   # Overbought - sell
            elif rsi <= 35:
                rsi_signal = 0.5    # Weakly oversold
            elif rsi >= 65:
                rsi_signal = -0.5   # Weakly overbought
        
        # 3. Bollinger Bands Signals (from Part 1)
        if all(key in indicators for key in ['BB_upper', 'BB_lower', 'BB_position']):
            bb_position = indicators['BB_position']
            
            if bb_position <= 0.1:
                bb_signal = 1.0     # Near lower band - buy
            elif bb_position >= 0.9:
                bb_signal = -1.0    # Near upper band - sell
            elif bb_position <= 0.2:
                bb_signal = 0.5     # Weakly oversold
            elif bb_position >= 0.8:
                bb_signal = -0.5    # Weakly overbought
        
        # 4. Ornstein-Uhlenbeck Process Signals (from Part 1)
        if all(key in indicators for key in ['OU_theta', 'OU_mu', 'OU_sigma']):
            ou_theta = indicators['OU_theta']
            ou_mu = indicators['OU_mu']
            ou_sigma = indicators['OU_sigma']
            
            if ou_sigma > 0 and ou_theta > 0:
                # Calculate mean reversion signal
                distance_from_mean = abs(current_price - ou_mu)
                normalized_distance = distance_from_mean / ou_sigma
                
                if normalized_distance > 2.0:
                    if current_price < ou_mu:
                        ou_signal = 1.0     # Far below mean - buy
                    else:
                        ou_signal = -1.0    # Far above mean - sell
                elif normalized_distance > 1.5:
                    if current_price < ou_mu:
                        ou_signal = 0.5     # Below mean - weak buy
                    else:
                        ou_signal = -0.5    # Above mean - weak sell
        
        # 5. Volume Confirmation (from Part 1)
        volume_signal = 1.0
        if len(self.volume_history[symbol]) >= 20:
            volumes = list(self.volume_history[symbol])
            avg_volume = np.mean(volumes[-20:])
            current_volume = volumes[-1]
            if current_volume > avg_volume * 1.2:
                volume_signal = 1.2  # High volume confirmation
        
        # Combine mean reversion signals using Part 1 methodology
        combined_strength = (
            z_score_signal * 0.3 +      # Z-Score (30%)
            rsi_signal * 0.25 +         # RSI (25%)
            bb_signal * 0.25 +          # Bollinger Bands (25%)
            ou_signal * 0.2             # Ornstein-Uhlenbeck (20%)
        ) * volume_signal
        
        # Quality filter for OU process
        ou_quality = 1.0
        if 'OU_theta' in indicators:
            ou_theta = indicators['OU_theta']
            if ou_theta > 0.01 and ou_theta < 1.0:  # Reasonable mean reversion speed
                ou_quality = 1.2
        
        combined_strength *= ou_quality
        
        # Determine signal type and confidence
        if combined_strength > 0.5:
            signal_type = 'buy'
            confidence = min(combined_strength, 1.0)
        elif combined_strength < -0.5:
            signal_type = 'sell'
            confidence = min(abs(combined_strength), 1.0)
        else:
            signal_type = 'hold'
            confidence = 0.0
        
        return StrategySignal(
            symbol=symbol,
            signal_type=signal_type,
            strength=abs(combined_strength),
            strategy_name="MeanReversion",
            timestamp=datetime.now(),
            confidence=confidence,
            additional_data={
                'z_score_signal': z_score_signal,
                'rsi_signal': rsi_signal,
                'bb_signal': bb_signal,
                'ou_signal': ou_signal,
                'volume_confirmation': volume_signal,
                'ou_quality': ou_quality
            }
        )
    
    def _combine_signals(self, trend_signal: StrategySignal, mean_rev_signal: StrategySignal) -> StrategySignal:
        """
        Combine trend following and mean reversion signals
        """
        # Weight assignment from Part 1 (60% trend, 40% mean reversion)
        trend_weight = 0.6
        mean_rev_weight = 0.4
        
        # Convert signal types to numeric values
        def signal_to_numeric(signal):
            if signal.signal_type == 'buy':
                return signal.strength
            elif signal.signal_type == 'sell':
                return -signal.strength
            else:
                return 0.0
        
        trend_numeric = signal_to_numeric(trend_signal)
        mean_rev_numeric = signal_to_numeric(mean_rev_signal)
        
        # Combine signals
        combined_strength = (trend_numeric * trend_weight + 
                           mean_rev_numeric * mean_rev_weight)
        
        combined_confidence = (trend_signal.confidence * trend_weight + 
                             mean_rev_signal.confidence * mean_rev_weight)
        
        # Determine final signal
        if combined_strength > 0.7:
            signal_type = 'buy'
        elif combined_strength < -0.7:
            signal_type = 'sell'
        else:
            signal_type = 'hold'
        
        return StrategySignal(
            symbol=trend_signal.symbol,
            signal_type=signal_type,
            strength=abs(combined_strength),
            strategy_name="CombinedTrendMeanReversion",
            timestamp=datetime.now(),
            confidence=combined_confidence,
            additional_data={
                'trend_component': trend_numeric,
                'mean_rev_component': mean_rev_numeric,
                'trend_confidence': trend_signal.confidence,
                'mean_rev_confidence': mean_rev_signal.confidence
            }
        )
    
    # Technical indicator calculation methods (from Part 1)
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return prices[-1] if prices else 0
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_macd(self, prices: List[float]) -> Tuple[float, float, float]:
        """Calculate MACD line, signal line, and histogram"""
        if len(prices) < self.macd_slow:
            return 0, 0, 0
        
        ema_fast = self._calculate_ema(prices, self.macd_fast)
        ema_slow = self._calculate_ema(prices, self.macd_slow)
        macd_line = ema_fast - ema_slow
        
        # For signal line, calculate EMA of MACD (simplified for demo)
        macd_signal_line = macd_line * 0.9  # Simplified
        macd_histogram = macd_line - macd_signal_line
        
        return macd_line, macd_signal_line, macd_histogram
    
    def _calculate_adx(self, prices: List[float]) -> float:
        """Calculate Average Directional Index (simplified)"""
        if len(prices) < self.adx_period + 10:
            return 25  # Default moderate trend strength
        
        # Simplified ADX calculation
        price_changes = np.diff(prices[-20:])
        up_moves = np.where(price_changes > 0, price_changes, 0)
        down_moves = np.where(price_changes < 0, -price_changes, 0)
        
        avg_up = np.mean(up_moves)
        avg_down = np.mean(down_moves)
        
        if avg_up + avg_down == 0:
            return 0
        
        dx = abs(avg_up - avg_down) / (avg_up + avg_down) * 100
        return min(dx, 100)
    
    def _calculate_rsi(self, prices: List[float], period: int) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50  # Neutral RSI
        
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_z_score(self, prices: List[float], window: int) -> float:
        """Calculate Z-Score"""
        if len(prices) < window:
            return 0
        
        recent_prices = prices[-window:]
        mean_price = np.mean(recent_prices)
        std_price = np.std(recent_prices)
        
        if std_price == 0:
            return 0
        
        return (prices[-1] - mean_price) / std_price
    
    def _calculate_bollinger_bands(self, prices: List[float]) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        if len(prices) < self.bollinger_period:
            current_price = prices[-1]
            return current_price * 1.02, current_price, current_price * 0.98
        
        recent_prices = prices[-self.bollinger_period:]
        middle = np.mean(recent_prices)
        std = np.std(recent_prices)
        
        upper = middle + (self.bollinger_std * std)
        lower = middle - (self.bollinger_std * std)
        
        return upper, middle, lower
    
    def _calculate_ou_process(self, prices: List[float]) -> Tuple[float, float, float]:
        """Calculate Ornstein-Uhlenbeck process parameters (simplified)"""
        if len(prices) < self.ou_window:
            return 0.1, prices[-1], np.std(prices[-10:]) if len(prices) >= 10 else 1.0
        
        # Simplified OU parameter estimation
        recent_prices = prices[-self.ou_window:]
        
        # Mean reversion level (mu)
        mu = np.mean(recent_prices)
        
        # Volatility (sigma)
        sigma = np.std(recent_prices)
        
        # Mean reversion speed (theta) - simplified calculation
        price_changes = np.diff(recent_prices)
        mean_deviation = np.mean([abs(p - mu) for p in recent_prices])
        
        if mean_deviation > 0:
            theta = 1.0 / mean_deviation  # Simplified theta calculation
        else:
            theta = 0.1
        
        # Cap theta to reasonable values
        theta = max(0.01, min(theta, 2.0))
        
        return theta, mu, sigma

class TradingOrchestrator:
    """Main orchestrator coordinating all trading components"""
    
    def __init__(self):
        # Core configuration
        self.api_key = "CKWWVBPP75ZM7BGWT7C7"
        self.secret_key = "8TqSb4XyuTodeFfZU6pSeDIBfMnU88R9oNiGep1E"
        self.account_id = "2e363eac-3981-351a-afbb-0322a4540912"
        self.semiconductor_symbols = {'NVDA', 'AMD', 'INTC', 'TSM', 'AVGO'}
        
        # Initialize components (would import from previous sections)
        self.api_client = AlpacaRESTClient()
        self.order_manager = OrderManager(self.api_client, self.semiconductor_symbols)
        self.data_validator = MarketDataValidator(list(self.semiconductor_symbols))
        self.websocket_client = AlpacaWebSocketClient(self.api_key, self.secret_key, self.data_validator)
        self.db_manager = DatabaseManager("live_trading.db")
        self.performance_analyzer = PerformanceAnalyzer(self.db_manager)
        self.risk_dashboard = RiskDashboard(self.db_manager, self.performance_analyzer)
        
        self.strategy_engine = StrategyEngine(list(self.semiconductor_symbols))
        
        # Trading state
        self.is_running = False
        self.emergency_stop_triggered = False
        self.last_heartbeat = datetime.now()
        
        # Event queues
        self.market_data_queue = queue.Queue(maxsize=1000)
        self.signal_queue = queue.Queue(maxsize=100)
        self.order_queue = queue.Queue(maxsize=100)
        
        # Threading
        self.threads = {}
        self.shutdown_event = threading.Event()
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.total_trades_today = 0
        self.start_of_day_capital = 50000.0
        
        # Risk limits
        self.max_daily_loss = 2500.0  # $2,500 max daily loss
        self.max_position_size = 0.15  # 15% max per position
        self.max_daily_trades = 20
        
    def initialize_system(self) -> bool:
        """Initialize all system components"""
        try:
            logger.info("Initializing trading system components...")
            
            # Test API connection
            account_response = self.api_client.get_account_info()
            if not account_response.success:
                logger.error("Failed to connect to Alpaca API")
                return False
            
            # Setup market data callbacks
            self.websocket_client.add_data_callback(self._on_market_data)
            self.websocket_client.add_error_callback(self._on_market_data_error)
            
            # Start order monitoring
            self.order_manager.start_monitoring()
            
            # Setup signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            logger.info("System initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    def start_trading(self):
        """Start the complete trading system"""
        if not self.initialize_system():
            logger.error("Failed to initialize system")
            return False
        
        self.is_running = True
        self.start_of_day_capital = 50000.0  # Would get from API
        
        logger.info("Starting live trading system...")
        
        # Start all background threads
        self._start_market_data_thread()
        self._start_signal_processing_thread()
        self._start_order_execution_thread()
        self._start_risk_monitoring_thread()
        self._start_performance_tracking_thread()
        
        # Connect to market data
        self.websocket_client.connect()
        
        logger.info("Live trading system started successfully")
        return True
    
    def stop_trading(self):
        """Stop the trading system gracefully"""
        logger.info("Stopping trading system...")
        
        self.is_running = False
        self.shutdown_event.set()
        
        # Cancel all open orders
        self.order_manager.emergency_cancel_all()
        
        # Disconnect market data
        self.websocket_client.disconnect()
        
        # Stop monitoring
        self.order_manager.stop_monitoring()
        
        # Wait for threads to finish
        for thread_name, thread in self.threads.items():
            if thread.is_alive():
                logger.info(f"Waiting for {thread_name} thread to finish...")
                thread.join(timeout=5)
        
        logger.info("Trading system stopped")
    
    def emergency_stop(self):
        """Emergency stop - immediate halt of all trading"""
        logger.critical("EMERGENCY STOP ACTIVATED")
        self.emergency_stop_triggered = True
        
        # Cancel all orders immediately
        self.order_manager.emergency_cancel_all()
        
        # Stop all trading
        self.stop_trading()
        
        logger.critical("Emergency stop completed")
    
    def _start_market_data_thread(self):
        """Start market data processing thread"""
        def market_data_processor():
            while self.is_running and not self.shutdown_event.is_set():
                try:
                    # Process market data from queue
                    if not self.market_data_queue.empty():
                        data_point = self.market_data_queue.get_nowait()
                        self.strategy_engine.update_market_data(data_point)
                        
                        # Check for new signals
                        symbol = data_point.symbol
                        if symbol in self.strategy_engine.signals:
                            signal = self.strategy_engine.signals[symbol]
                            if signal.signal_type != 'hold':
                                self.signal_queue.put(signal)
                    
                    time.sleep(0.1)  # 100ms processing interval
                    
                except Exception as e:
                    logger.error(f"Error in market data processor: {e}")
                    time.sleep(1)
        
        thread = threading.Thread(target=market_data_processor, daemon=True)
        thread.start()
        self.threads['market_data'] = thread
    
    def _start_signal_processing_thread(self):
        """Start signal processing thread"""
        def signal_processor():
            while self.is_running and not self.shutdown_event.is_set():
                try:
                    if not self.signal_queue.empty():
                        signal = self.signal_queue.get_nowait()
                        
                        # Check if we should act on this signal
                        if self._should_execute_signal(signal):
                            order_request = self._create_order_from_signal(signal)
                            if order_request:
                                self.order_queue.put(order_request)
                    
                    time.sleep(0.5)  # 500ms signal processing interval
                    
                except Exception as e:
                    logger.error(f"Error in signal processor: {e}")
                    time.sleep(1)
        
        thread = threading.Thread(target=signal_processor, daemon=True)
        thread.start()
        self.threads['signal_processing'] = thread
    
    def _start_order_execution_thread(self):
        """Start order execution thread"""
        def order_executor():
            while self.is_running and not self.shutdown_event.is_set():
                try:
                    if not self.order_queue.empty():
                        order_request = self.order_queue.get_nowait()
                        
                        # Execute order with safety checks
                        if self._pre_trade_risk_check(order_request):
                            success, message, order_id = self.order_manager.submit_order(
                                order_request, 
                                current_price=450.0,  # Would get from market data
                                account_value=50000.0   # Would get from API
                            )
                            
                            # Placeholder for actual execution
                            success, message, order_id = True, "Order simulated", "SIM123"
                            
                            if success:
                                self.total_trades_today += 1
                                logger.info(f"Order executed: {message}")
                                
                                # Record trade
                                trade_record = TradeRecord(
                                    trade_id=order_id,
                                    symbol=order_request.symbol,
                                    side=order_request.side,
                                    quantity=order_request.quantity,
                                    fill_price=450.0,  # Would get actual fill price
                                    fill_time=datetime.now(),
                                    strategy_id=order_request.strategy_id,
                                    order_type=order_request.order_type
                                )
                                self.db_manager.save_trade(trade_record)
                            else:
                                logger.warning(f"Order failed: {message}")
                        else:
                            logger.warning(f"Order blocked by risk check: {order_request.symbol}")
                    
                    time.sleep(1)  # 1 second order processing interval
                    
                except Exception as e:
                    logger.error(f"Error in order executor: {e}")
                    time.sleep(2)
        
        thread = threading.Thread(target=order_executor, daemon=True)
        thread.start()
        self.threads['order_execution'] = thread
    
    def _start_risk_monitoring_thread(self):
        """Start risk monitoring thread"""
        def risk_monitor():
            while self.is_running and not self.shutdown_event.is_set():
                try:
                    # Check daily P&L
                    current_pnl = self._calculate_current_pnl()
                    
                    if current_pnl < -self.max_daily_loss:
                        logger.critical(f"Daily loss limit breached: ${current_pnl:.2f}")
                        self.emergency_stop()
                        break
                    
                    # Check trade count
                    if self.total_trades_today >= self.max_daily_trades:
                        logger.warning("Daily trade limit reached")
                        # Could pause trading instead of emergency stop
                    
                    # Check connection health
                    if not self.websocket_client.check_connection_health():
                        logger.warning("Market data connection unhealthy")
                    
                    self.last_heartbeat = datetime.now()
                    time.sleep(30)  # 30-second risk monitoring interval
                    
                except Exception as e:
                    logger.error(f"Error in risk monitor: {e}")
                    time.sleep(60)
        
        thread = threading.Thread(target=risk_monitor, daemon=True)
        thread.start()
        self.threads['risk_monitoring'] = thread
    
    def _start_performance_tracking_thread(self):
        """Start performance tracking thread"""
        def performance_tracker():
            while self.is_running and not self.shutdown_event.is_set():
                try:
                    # Update daily P&L
                    self.daily_pnl = self._calculate_current_pnl()
                    
                    # Log performance metrics every 5 minutes
                    self._log_performance_metrics()
                    
                    time.sleep(300)  # 5-minute performance tracking interval
                    
                except Exception as e:
                    logger.error(f"Error in performance tracker: {e}")
                    time.sleep(60)
        
        thread = threading.Thread(target=performance_tracker, daemon=True)
        thread.start()
        self.threads['performance_tracking'] = thread
    
    def _should_execute_signal(self, signal: StrategySignal) -> bool:
        """Determine if signal should result in trade execution"""
        # Check signal strength and confidence
        if signal.strength < 0.7 or signal.confidence < 0.6:
            return False
        
        # Check if we're in emergency stop
        if self.emergency_stop_triggered:
            return False
        
        # Check daily limits
        if self.total_trades_today >= self.max_daily_trades:
            return False
        
        # Check if symbol is still valid
        if signal.symbol not in self.semiconductor_symbols:
            return False
        
        # Check signal freshness (no older than 1 minute)
        signal_age = (datetime.now() - signal.timestamp).total_seconds()
        if signal_age > 60:
            return False
        
        return True
    
    def _create_order_from_signal(self, signal: StrategySignal):
        """Create order request from strategy signal"""
        # Calculate position size based on confidence and risk management
        base_position_size = 100  # Base 100 shares
        confidence_multiplier = signal.confidence
        position_size = int(base_position_size * confidence_multiplier)
        
        # Set prices far from market to avoid execution (as requested)
        current_price = 450.0  # Would get from market data
        
        if signal.signal_type == 'buy':
            # Set limit price 5% below market
            limit_price = current_price * 0.95
            side = 'buy'
        elif signal.signal_type == 'sell':
            # Set limit price 5% above market
            limit_price = current_price * 1.05
            side = 'sell'
        else:
            return None
        
        # Create order request (would import from order_management_system)
        return OrderRequest(
            symbol=signal.symbol,
            quantity=position_size,
            side=side,
            order_type='limit',
            limit_price=limit_price,
            strategy_id=signal.strategy_name
        )
        
        # Placeholder return
        return {
            'symbol': signal.symbol,
            'quantity': position_size,
            'side': side,
            'order_type': 'limit',
            'limit_price': limit_price,
            'strategy_id': signal.strategy_name
        }
    
    def _pre_trade_risk_check(self, order_request) -> bool:
        """Pre-trade risk validation"""
        # Check position concentration
        order_value = order_request['quantity'] * 450.0  # Would use actual price
        max_position_value = self.start_of_day_capital * self.max_position_size
        
        if order_value > max_position_value:
            return False
        
        # Check daily loss limit buffer
        if self.daily_pnl < -self.max_daily_loss * 0.8:  # 80% of limit
            return False
        
        return True
    
    def _calculate_current_pnl(self) -> float:
        """Calculate current day P&L"""
        # Would calculate from actual positions and trades
        # For now, return simulated P&L
        return self.daily_pnl
    
    def _log_performance_metrics(self):
        """Log current performance metrics"""
        pnl_percent = (self.daily_pnl / self.start_of_day_capital) * 100
        
        logger.info(f"Performance Update:")
        logger.info(f"  Daily P&L: ${self.daily_pnl:.2f} ({pnl_percent:.2f}%)")
        logger.info(f"  Trades Today: {self.total_trades_today}")
        logger.info(f"  Queue Sizes: Data={self.market_data_queue.qsize()}, "
                   f"Signals={self.signal_queue.qsize()}, Orders={self.order_queue.qsize()}")
    
    def _on_market_data(self, data_point):
        """Callback for market data updates"""
        try:
            self.market_data_queue.put_nowait(data_point)
        except queue.Full:
            logger.warning("Market data queue full, dropping data point")
    
    def _on_market_data_error(self, error_message: str):
        """Callback for market data errors"""
        logger.error(f"Market data error: {error_message}")
        
        # If critical error, consider emergency stop
        if "connection" in error_message.lower():
            logger.critical("Market data connection lost - emergency stop may be needed")
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop_trading()
        sys.exit(0)
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        return {
            'is_running': self.is_running,
            'emergency_stop': self.emergency_stop_triggered,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_percent': (self.daily_pnl / self.start_of_day_capital) * 100,
            'trades_today': self.total_trades_today,
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'queue_sizes': {
                'market_data': self.market_data_queue.qsize(),
                'signals': self.signal_queue.qsize(),
                'orders': self.order_queue.qsize()
            },
            'active_threads': [name for name, thread in self.threads.items() if thread.is_alive()],
            'symbols_tracked': list(self.semiconductor_symbols)
        }
    
    def print_status_dashboard(self):
        """Print formatted status dashboard"""
        status = self.get_system_status()
        
        print(f"\n{'='*80}")
        print(f"LIVE TRADING SYSTEM STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        print(f"System Status: {'RUNNING' if status['is_running'] else 'STOPPED'}")
        if status['emergency_stop']:
            print(f"EMERGENCY STOP ACTIVE")
        
        print(f"\nPerformance:")
        print(f"  Daily P&L: ${status['daily_pnl']:,.2f} ({status['daily_pnl_percent']:+.2f}%)")
        print(f"  Trades Today: {status['trades_today']}")
        
        print(f"\nSystem Health:")
        print(f"  Last Heartbeat: {status['last_heartbeat']}")
        print(f"  Active Threads: {len(status['active_threads'])}")
        
        print(f"\nQueue Status:")
        for queue_name, size in status['queue_sizes'].items():
            print(f"  {queue_name.title()}: {size} items")
        
        print(f"\nSymbols: {', '.join(status['symbols_tracked'])}")
        print(f"{'='*80}")

def main():
    """Main execution function"""
    print("Starting Semiconductor Live Trading System...")
    print("Using Alpaca Broker API with AL_JAKIUR_RAHMAN strategies")
    
    # Initialize orchestrator
    orchestrator = TradingOrchestrator()
    
    try:
        # Start trading system
        if orchestrator.start_trading():
            print("Trading system started successfully")
            
            # Run main loop
            while orchestrator.is_running:
                try:
                    # Print status every 60 seconds
                    orchestrator.print_status_dashboard()
                    time.sleep(60)
                    
                except KeyboardInterrupt:
                    print("\nKeyboard interrupt received")
                    break
        else:
            print("Failed to start trading system")
    
    except Exception as e:
        print(f"System error: {e}")
        orchestrator.emergency_stop()
    
    finally:
        # Ensure clean shutdown
        orchestrator.stop_trading()
        print("Trading system shutdown complete")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_trading.log'),
        logging.StreamHandler()
    ]
)

# RUNNING MAIN
main()