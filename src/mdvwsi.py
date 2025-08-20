# Market Data Validation & WebSocket Integration
import websocket
import json
import threading
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
from collections import deque, defaultdict
import logging
import queue
import ssl
from enum import Enum

logger = logging.getLogger(__name__)

class DataQualityLevel(Enum):
    """Data quality assessment levels"""
    EXCELLENT = "excellent"
    GOOD = "good" 
    FAIR = "fair"
    POOR = "poor"
    INVALID = "invalid"

@dataclass
class MarketDataPoint:
    """Standardized market data point"""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    source: str = "unknown"
    quality: DataQualityLevel = DataQualityLevel.GOOD

@dataclass
class DataQualityReport:
    """Data quality assessment report"""
    symbol: str
    total_points: int
    valid_points: int
    invalid_points: int
    missing_data_periods: int
    price_anomalies: int
    volume_anomalies: int
    stale_data_count: int
    last_update: datetime
    overall_quality: DataQualityLevel

class MarketDataValidator:
    """Comprehensive market data validation system"""
    
    def __init__(self, semiconductor_symbols: List[str]):
        self.symbols = semiconductor_symbols
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.last_prices: Dict[str, float] = {}
        self.last_update: Dict[str, datetime] = {}
        
        # Validation thresholds
        self.max_price_change_percent = 0.15  # 15% max change per minute
        self.max_volume_multiplier = 10.0     # 10x average volume
        self.min_price = 0.01                 # Minimum valid price
        self.max_stale_seconds = 300          # 5 minutes stale data threshold
        
        # Quality tracking
        self.quality_reports: Dict[str, DataQualityReport] = {}
    
    def validate_price_data(self, symbol: str, price: float, timestamp: datetime) -> Tuple[bool, str]:
        """Validate price data for anomalies"""
        
        # Basic price validation
        if price <= self.min_price:
            return False, f"Price too low: {price}"
        
        if not np.isfinite(price):
            return False, f"Invalid price value: {price}"
        
        # Historical price validation
        if symbol in self.last_prices:
            last_price = self.last_prices[symbol]
            price_change = abs(price - last_price) / last_price
            
            if price_change > self.max_price_change_percent:
                return False, f"Excessive price change: {price_change:.2%} from {last_price} to {price}"
        
        # Futures vs spot validation (semiconductor-specific)
        if self._is_futures_symbol(symbol):
            spot_price = self._get_spot_price(symbol)
            if spot_price and price < spot_price * 0.8:  # Futures significantly below spot
                return False, f"Futures price {price} too low vs spot {spot_price}"
        
        return True, "Price validation passed"
    
    def validate_volume_data(self, symbol: str, volume: int, timestamp: datetime) -> Tuple[bool, str]:
        """Validate volume data for anomalies"""
        
        # Basic volume validation
        if volume < 0:
            return False, f"Negative volume: {volume}"
        
        if not np.isfinite(volume):
            return False, f"Invalid volume value: {volume}"
        
        # Historical volume validation
        if len(self.volume_history[symbol]) > 10:
            avg_volume = np.mean(self.volume_history[symbol])
            
            if volume > avg_volume * self.max_volume_multiplier:
                return False, f"Excessive volume: {volume} vs avg {avg_volume:.0f}"
        
        return True, "Volume validation passed"
    
    def validate_bid_ask_spread(self, symbol: str, bid: float, ask: float) -> Tuple[bool, str]:
        """Validate bid-ask spread for reasonableness"""
        
        if bid >= ask:
            return False, f"Invalid spread: bid {bid} >= ask {ask}"
        
        mid_price = (bid + ask) / 2
        spread_percent = (ask - bid) / mid_price
        
        # Semiconductor stocks typically have tight spreads
        max_spread_percent = 0.02  # 2% max spread
        
        if spread_percent > max_spread_percent:
            return False, f"Excessive spread: {spread_percent:.2%}"
        
        return True, "Spread validation passed"
    
    def check_data_freshness(self, symbol: str, timestamp: datetime) -> Tuple[bool, str]:
        """Check if data is fresh (not stale)"""
        
        now = datetime.now()
        age_seconds = (now - timestamp).total_seconds()
        
        if age_seconds > self.max_stale_seconds:
            return False, f"Stale data: {age_seconds:.0f} seconds old"
        
        return True, "Data freshness check passed"
    
    def validate_market_data(self, data_point: MarketDataPoint) -> Tuple[bool, str, DataQualityLevel]:
        """Comprehensive market data validation"""
        
        validation_errors = []
        
        # Price validation
        price_valid, price_msg = self.validate_price_data(
            data_point.symbol, data_point.price, data_point.timestamp
        )
        if not price_valid:
            validation_errors.append(f"Price: {price_msg}")
        
        # Volume validation
        volume_valid, volume_msg = self.validate_volume_data(
            data_point.symbol, data_point.volume, data_point.timestamp
        )
        if not volume_valid:
            validation_errors.append(f"Volume: {volume_msg}")
        
        # Bid-ask validation
        if data_point.bid and data_point.ask:
            spread_valid, spread_msg = self.validate_bid_ask_spread(
                data_point.symbol, data_point.bid, data_point.ask
            )
            if not spread_valid:
                validation_errors.append(f"Spread: {spread_msg}")
        
        # Freshness check
        fresh_valid, fresh_msg = self.check_data_freshness(
            data_point.symbol, data_point.timestamp
        )
        if not fresh_valid:
            validation_errors.append(f"Freshness: {fresh_msg}")
        
        # Determine overall quality
        if validation_errors:
            quality = DataQualityLevel.POOR if len(validation_errors) > 2 else DataQualityLevel.FAIR
            return False, "; ".join(validation_errors), quality
        else:
            quality = DataQualityLevel.EXCELLENT if fresh_valid and price_valid and volume_valid else DataQualityLevel.GOOD
            return True, "All validations passed", quality
    
    def update_data_history(self, data_point: MarketDataPoint):
        """Update historical data for validation"""
        symbol = data_point.symbol
        
        self.price_history[symbol].append(data_point.price)
        self.volume_history[symbol].append(data_point.volume)
        self.last_prices[symbol] = data_point.price
        self.last_update[symbol] = data_point.timestamp
    
    def _is_futures_symbol(self, symbol: str) -> bool:
        """Check if symbol represents futures contract"""
        # Simple heuristic - futures symbols often have month/year codes
        return len(symbol) > 5 and any(char.isdigit() for char in symbol[-2:])
    
    def _get_spot_price(self, futures_symbol: str) -> Optional[float]:
        """Get corresponding spot price for futures validation"""
        # Extract base symbol (simplified)
        base_symbol = futures_symbol[:4] if len(futures_symbol) > 4 else futures_symbol
        return self.last_prices.get(base_symbol)
    
    def generate_quality_report(self, symbol: str) -> DataQualityReport:
        """Generate data quality report for symbol"""
        # This would typically analyze recent data history
        total_points = len(self.price_history[symbol])
        
        report = DataQualityReport(
            symbol=symbol,
            total_points=total_points,
            valid_points=max(0, total_points - 5),  # Simplified
            invalid_points=min(5, total_points),
            missing_data_periods=0,
            price_anomalies=0,
            volume_anomalies=0,
            stale_data_count=0,
            last_update=self.last_update.get(symbol, datetime.now()),
            overall_quality=DataQualityLevel.GOOD
        )
        
        self.quality_reports[symbol] = report
        return report

class AlpacaWebSocketClient:
    """Alpaca WebSocket client with robust connection handling"""
    
    def __init__(self, api_key: str, secret_key: str, data_validator: MarketDataValidator):
        self.api_key = api_key
        self.secret_key = secret_key
        self.data_validator = data_validator
        
        # WebSocket configuration
        self.ws_url = "wss://stream.data.alpaca.markets/v2/iex"
        self.ws = None
        self.is_connected = False
        self.should_reconnect = True
        
        # Data handling
        self.data_queue = queue.Queue(maxsize=10000)
        self.data_callbacks: List[Callable[[MarketDataPoint], None]] = []
        self.error_callbacks: List[Callable[[str], None]] = []
        
        # Connection management
        self.connection_thread = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 5
        
        # Monitoring
        self.last_heartbeat = None
        self.heartbeat_interval = 30
        self.stale_connection_threshold = 120
        
        # Statistics
        self.messages_received = 0
        self.valid_messages = 0
        self.invalid_messages = 0
        self.connection_uptime_start = None
    
    def add_data_callback(self, callback: Callable[[MarketDataPoint], None]):
        """Add callback for validated data"""
        self.data_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[str], None]):
        """Add callback for errors"""
        self.error_callbacks.append(callback)
    
    def on_open(self, ws):
        """WebSocket connection opened"""
        logger.info("WebSocket connection established")
        self.is_connected = True
        self.reconnect_attempts = 0
        self.connection_uptime_start = datetime.now()
        self.last_heartbeat = datetime.now()
        
        # Authenticate
        auth_message = {
            "action": "auth",
            "key": self.api_key,
            "secret": self.secret_key
        }
        ws.send(json.dumps(auth_message))
        
        # Subscribe to semiconductor symbols
        subscribe_message = {
            "action": "subscribe",
            "trades": self.data_validator.symbols,
            "quotes": self.data_validator.symbols
        }
        ws.send(json.dumps(subscribe_message))
    
    def on_message(self, ws, message):
        """Process incoming WebSocket message"""
        try:
            self.messages_received += 1
            self.last_heartbeat = datetime.now()
            
            data = json.loads(message)
            
            # Handle different message types
            if isinstance(data, list):
                for item in data:
                    self._process_single_message(item)
            else:
                self._process_single_message(data)
                
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
            for callback in self.error_callbacks:
                callback(f"Message processing error: {e}")
    
    def _process_single_message(self, message: Dict):
        """Process individual market data message"""
        try:
            msg_type = message.get('T')
            symbol = message.get('S')
            
            if not symbol or symbol not in self.data_validator.symbols:
                return
            
            # Process trade messages
            if msg_type == 't':  # Trade
                price = float(message.get('p', 0))
                volume = int(message.get('s', 0))
                timestamp = datetime.fromisoformat(message.get('t', datetime.now().isoformat()))
                
                data_point = MarketDataPoint(
                    symbol=symbol,
                    timestamp=timestamp,
                    price=price,
                    volume=volume,
                    source="alpaca_websocket"
                )
                
                # Validate data
                is_valid, validation_msg, quality = self.data_validator.validate_market_data(data_point)
                
                if is_valid:
                    self.valid_messages += 1
                    data_point.quality = quality
                    
                    # Update validator history
                    self.data_validator.update_data_history(data_point)
                    
                    # Add to queue and notify callbacks
                    try:
                        self.data_queue.put_nowait(data_point)
                    except queue.Full:
                        logger.warning("Data queue full, dropping message")
                    
                    for callback in self.data_callbacks:
                        try:
                            callback(data_point)
                        except Exception as e:
                            logger.error(f"Error in data callback: {e}")
                else:
                    self.invalid_messages += 1
                    logger.warning(f"Invalid data for {symbol}: {validation_msg}")
            
            # Process quote messages
            elif msg_type == 'q':  # Quote
                bid = float(message.get('bp', 0))
                ask = float(message.get('ap', 0))
                timestamp = datetime.fromisoformat(message.get('t', datetime.now().isoformat()))
                
                # Validate bid-ask spread
                spread_valid, spread_msg = self.data_validator.validate_bid_ask_spread(symbol, bid, ask)
                
                if spread_valid:
                    # Create data point with quote info
                    mid_price = (bid + ask) / 2
                    data_point = MarketDataPoint(
                        symbol=symbol,
                        timestamp=timestamp,
                        price=mid_price,
                        volume=0,  # Quotes don't have volume
                        bid=bid,
                        ask=ask,
                        source="alpaca_websocket"
                    )
                    
                    for callback in self.data_callbacks:
                        try:
                            callback(data_point)
                        except Exception as e:
                            logger.error(f"Error in quote callback: {e}")
                else:
                    logger.warning(f"Invalid quote for {symbol}: {spread_msg}")
                    
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def on_error(self, ws, error):
        """WebSocket error handler"""
        logger.error(f"WebSocket error: {error}")
        self.is_connected = False
        
        for callback in self.error_callbacks:
            callback(f"WebSocket error: {error}")
        
        # Trigger reconnection
        if self.should_reconnect and self.reconnect_attempts < self.max_reconnect_attempts:
            self._schedule_reconnect()
    
    def on_close(self, ws, close_status_code, close_msg):
        """WebSocket connection closed"""
        logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
        self.is_connected = False
        
        if self.should_reconnect and self.reconnect_attempts < self.max_reconnect_attempts:
            self._schedule_reconnect()
    
    def _schedule_reconnect(self):
        """Schedule reconnection attempt"""
        self.reconnect_attempts += 1
        delay = min(self.reconnect_delay * (2 ** (self.reconnect_attempts - 1)), 60)
        
        logger.info(f"Scheduling reconnect attempt {self.reconnect_attempts} in {delay} seconds")
        
        def reconnect():
            time.sleep(delay)
            if self.should_reconnect:
                self.connect()
        
        threading.Thread(target=reconnect, daemon=True).start()
    
    def connect(self):
        """Establish WebSocket connection"""
        try:
            if self.is_connected:
                return
            
            logger.info("Connecting to Alpaca WebSocket...")
            
            # Create WebSocket with SSL context
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            # Start connection in thread
            self.connection_thread = threading.Thread(
                target=self.ws.run_forever,
                kwargs={'sslopt': {"cert_reqs": ssl.CERT_NONE}},
                daemon=True
            )
            self.connection_thread.start()
            
        except Exception as e:
            logger.error(f"Failed to connect WebSocket: {e}")
            for callback in self.error_callbacks:
                callback(f"Connection failed: {e}")
    
    def disconnect(self):
        """Disconnect WebSocket"""
        self.should_reconnect = False
        self.is_connected = False
        
        if self.ws:
            self.ws.close()
        
        logger.info("WebSocket disconnected")
    
    def check_connection_health(self) -> bool:
        """Check WebSocket connection health"""
        if not self.is_connected:
            return False
        
        if not self.last_heartbeat:
            return False
        
        time_since_heartbeat = (datetime.now() - self.last_heartbeat).total_seconds()
        
        if time_since_heartbeat > self.stale_connection_threshold:
            logger.warning(f"Stale WebSocket connection: {time_since_heartbeat}s since last message")
            return False
        
        return True
    
    def get_connection_stats(self) -> Dict:
        """Get connection statistics"""
        uptime = None
        if self.connection_uptime_start:
            uptime = (datetime.now() - self.connection_uptime_start).total_seconds()
        
        return {
            'is_connected': self.is_connected,
            'messages_received': self.messages_received,
            'valid_messages': self.valid_messages,
            'invalid_messages': self.invalid_messages,
            'validation_rate': self.valid_messages / max(1, self.messages_received),
            'reconnect_attempts': self.reconnect_attempts,
            'uptime_seconds': uptime,
            'last_heartbeat': self.last_heartbeat,
            'queue_size': self.data_queue.qsize()
        }

# Test the Market Data Validation System

print("Testing Market Data Validation System...")

# Initialize validator
semiconductor_symbols = ['NVDA', 'AMD', 'INTC', 'TSM', 'AVGO']
validator = MarketDataValidator(semiconductor_symbols)

# Test price validation
print("\n1. Testing price validation...")

# Valid price
is_valid, msg = validator.validate_price_data("NVDA", 450.0, datetime.now())
print(f"Valid price test: {is_valid} - {msg}")

# Invalid price (too low)
is_valid, msg = validator.validate_price_data("NVDA", 0.001, datetime.now())
print(f"Invalid low price test: {is_valid} - {msg}")

# Test volume validation
print("\n2. Testing volume validation...")

# Valid volume
is_valid, msg = validator.validate_volume_data("NVDA", 1000000, datetime.now())
print(f"Valid volume test: {is_valid} - {msg}")

# Invalid volume (negative)
is_valid, msg = validator.validate_volume_data("NVDA", -1000, datetime.now())
print(f"Invalid negative volume test: {is_valid} - {msg}")

# Test bid-ask validation
print("\n3. Testing bid-ask spread validation...")

# Valid spread
is_valid, msg = validator.validate_bid_ask_spread("NVDA", 449.50, 450.50)
print(f"Valid spread test: {is_valid} - {msg}")

# Invalid spread (bid >= ask)
is_valid, msg = validator.validate_bid_ask_spread("NVDA", 450.50, 449.50)
print(f"Invalid spread test: {is_valid} - {msg}")

# Test full data validation
print("\n4. Testing full data validation...")

test_data = MarketDataPoint(
    symbol="NVDA",
    timestamp=datetime.now(),
    price=450.0,
    volume=1000000,
    bid=449.50,
    ask=450.50,
    source="test"
)

is_valid, msg, quality = validator.validate_market_data(test_data)
print(f"Full validation test: {is_valid} - {msg} - Quality: {quality}")

# Initialize WebSocket client (won't connect without real credentials)
print("\n5. Initializing WebSocket client...")
ws_client = AlpacaWebSocketClient("test_key", "test_secret", validator)

# Test connection stats
stats = ws_client.get_connection_stats()
print(f"Connection stats: {stats}")

print("Market Data Validation System testing complete.")