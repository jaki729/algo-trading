# Advanced Order Management & Position Tracking System
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid
import threading
import time
import logging
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)

@dataclass
class OrderRequest:
    """Order request with validation"""
    symbol: str
    quantity: int
    side: str  # 'buy' or 'sell'
    order_type: str = "market"
    time_in_force: str = "day"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_price: Optional[float] = None
    trail_percent: Optional[float] = None
    client_order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    strategy_id: str = "default"
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate order parameters"""
        self.symbol = self.symbol.upper()
        if self.side not in ['buy', 'sell']:
            raise ValueError("Side must be 'buy' or 'sell'")
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")

@dataclass
class Position:
    """Position tracking with P&L calculation"""
    symbol: str
    quantity: int
    side: str
    avg_price: float
    market_price: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)
    
    @property
    def market_value(self) -> float:
        """Current market value of position"""
        return abs(self.quantity) * self.market_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L"""
        if self.side == 'long':
            return self.quantity * (self.market_price - self.avg_price)
        else:  # short
            return self.quantity * (self.avg_price - self.market_price)
    
    @property
    def unrealized_pnl_percent(self) -> float:
        """Unrealized P&L as percentage"""
        if self.avg_price == 0:
            return 0.0
        return (self.unrealized_pnl / (abs(self.quantity) * self.avg_price)) * 100

class OrderValidator:
    """Comprehensive order validation system"""
    
    def __init__(self, semiconductor_symbols: Set[str]):
        self.valid_symbols = semiconductor_symbols
        self.min_order_value = 1.0  # Minimum $1 order
        self.max_order_value = 50000.0  # Maximum order size
        self.max_position_concentration = 0.3  # 30% max per symbol
    
    def validate_order(self, order: OrderRequest, current_price: float, 
                      account_value: float, current_positions: Dict[str, Position]) -> Tuple[bool, str]:
        """Comprehensive order validation"""
        
        # Symbol validation
        if order.symbol not in self.valid_symbols:
            return False, f"Invalid symbol: {order.symbol}. Must be semiconductor stock."
        
        # Price validation
        if current_price <= 0:
            return False, f"Invalid current price: {current_price}"
        
        # Order value validation
        order_value = order.quantity * current_price
        if order_value < self.min_order_value:
            return False, f"Order value ${order_value:.2f} below minimum ${self.min_order_value}"
        
        if order_value > self.max_order_value:
            return False, f"Order value ${order_value:.2f} exceeds maximum ${self.max_order_value}"
        
        # Limit price validation (set far from market to avoid execution)
        if order.order_type == "limit":
            if order.limit_price is None:
                return False, "Limit price required for limit orders"
            
            # Ensure limit prices are far from market (5% minimum)
            price_diff_percent = abs(order.limit_price - current_price) / current_price
            if price_diff_percent < 0.05:  # 5% minimum difference
                return False, f"Limit price too close to market. Must be >5% away. Current: {current_price:.2f}, Limit: {order.limit_price:.2f}"
        
        # Stop price validation
        if order.order_type in ["stop", "stop_limit"]:
            if order.stop_price is None:
                return False, f"Stop price required for {order.order_type} orders"
            
            # Stop price logic validation
            if order.side == "buy" and order.stop_price <= current_price:
                return False, "Buy stop price must be above current market price"
            elif order.side == "sell" and order.stop_price >= current_price:
                return False, "Sell stop price must be below current market price"
        
        # Position concentration check
        current_position_value = 0
        if order.symbol in current_positions:
            current_position_value = current_positions[order.symbol].market_value
        
        new_position_value = current_position_value + order_value
        concentration = new_position_value / account_value
        
        if concentration > self.max_position_concentration:
            return False, f"Position concentration {concentration:.1%} exceeds maximum {self.max_position_concentration:.1%}"
        
        return True, "Order validation passed"

class OrderManager:
    """Advanced order management with tracking and reconciliation"""
    
    def __init__(self, api_client, semiconductor_symbols: Set[str]):
        self.api_client = api_client
        self.validator = OrderValidator(semiconductor_symbols)
        
        # Order tracking
        self.pending_orders: Dict[str, OrderRequest] = {}
        self.submitted_orders: Dict[str, Dict] = {}  # API order responses
        self.filled_orders: List[Dict] = []
        self.rejected_orders: List[Dict] = []
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.position_lock = threading.Lock()
        
        # Monitoring
        self.order_update_thread = None
        self.is_monitoring = False
        self.last_position_update = None
        
        # Risk metrics
        self.daily_trades = 0
        self.daily_volume = 0.0
        self.max_daily_trades = 50
        self.max_daily_volume = 25000.0  # $25k daily volume limit
    
    def submit_order(self, order_request: OrderRequest, current_price: float, 
                    account_value: float) -> Tuple[bool, str, Optional[str]]:
        """Submit order with comprehensive validation and tracking"""
        
        # Daily limits check
        if self.daily_trades >= self.max_daily_trades:
            return False, "Daily trade limit reached", None
        
        order_value = order_request.quantity * current_price
        if self.daily_volume + order_value > self.max_daily_volume:
            return False, "Daily volume limit would be exceeded", None
        
        # Validate order
        is_valid, validation_msg = self.validator.validate_order(
            order_request, current_price, account_value, self.positions
        )
        
        if not is_valid:
            return False, validation_msg, None
        
        # Submit to API
        try:
            api_response = self.api_client.place_order(
                symbol=order_request.symbol,
                qty=order_request.quantity,
                side=order_request.side,
                order_type=order_request.order_type,
                time_in_force=order_request.time_in_force,
                limit_price=order_request.limit_price,
                stop_price=order_request.stop_price,
                trail_price=order_request.trail_price,
                trail_percent=order_request.trail_percent
            )
            
            if api_response.success:
                order_data = api_response.data
                order_id = order_data.get('id')
                
                # Track order
                self.pending_orders[order_request.client_order_id] = order_request
                self.submitted_orders[order_id] = order_data
                
                # Update daily counters
                self.daily_trades += 1
                self.daily_volume += order_value
                
                logger.info(f"Order submitted successfully: {order_id} for {order_request.symbol}")
                return True, f"Order submitted: {order_id}", order_id
            else:
                logger.error(f"Order submission failed: {api_response.error}")
                return False, f"API Error: {api_response.error}", None
                
        except Exception as e:
            logger.error(f"Exception during order submission: {e}")
            return False, f"Submission error: {str(e)}", None
    
    def cancel_order(self, order_id: str) -> Tuple[bool, str]:
        """Cancel order with verification"""
        try:
            api_response = self.api_client.cancel_order(order_id)
            
            if api_response.success:
                logger.info(f"Order cancelled successfully: {order_id}")
                return True, "Order cancelled"
            else:
                logger.error(f"Order cancellation failed: {api_response.error}")
                return False, f"Cancellation failed: {api_response.error}"
                
        except Exception as e:
            logger.error(f"Exception during order cancellation: {e}")
            return False, f"Cancellation error: {str(e)}"
    
    def emergency_cancel_all(self) -> Tuple[bool, str]:
        """Emergency cancellation of all orders"""
        try:
            api_response = self.api_client.cancel_all_orders()
            
            if api_response.success:
                logger.critical("All orders cancelled in emergency stop")
                self.pending_orders.clear()
                return True, "All orders cancelled"
            else:
                logger.error(f"Emergency cancellation failed: {api_response.error}")
                return False, f"Emergency cancellation failed: {api_response.error}"
                
        except Exception as e:
            logger.error(f"Exception during emergency cancellation: {e}")
            return False, f"Emergency cancellation error: {str(e)}"
    
    def update_positions(self) -> bool:
        """Update positions from API with verification"""
        try:
            api_response = self.api_client.get_positions()
            
            if not api_response.success:
                logger.error(f"Failed to update positions: {api_response.error}")
                return False
            
            positions_data = api_response.data
            
            with self.position_lock:
                # Clear current positions
                self.positions.clear()
                
                # Update with API data
                for pos_data in positions_data:
                    symbol = pos_data.get('symbol')
                    qty = int(pos_data.get('qty', 0))
                    
                    if qty != 0:  # Only track non-zero positions
                        side = 'long' if qty > 0 else 'short'
                        avg_price = float(pos_data.get('avg_entry_price', 0))
                        market_price = float(pos_data.get('market_value', 0)) / abs(qty) if qty != 0 else 0
                        
                        position = Position(
                            symbol=symbol,
                            quantity=abs(qty),
                            side=side,
                            avg_price=avg_price,
                            market_price=market_price,
                            last_update=datetime.now()
                        )
                        
                        self.positions[symbol] = position
                
                self.last_position_update = datetime.now()
                logger.info(f"Updated {len(self.positions)} positions")
                return True
                
        except Exception as e:
            logger.error(f"Exception updating positions: {e}")
            return False
    
    def update_orders(self) -> Dict[str, int]:
        """Update order statuses and reconcile"""
        status_counts = defaultdict(int)
        
        try:
            # Get all orders
            api_response = self.api_client.get_orders()
            if not api_response.success:
                logger.error(f"Failed to update orders: {api_response.error}")
                return status_counts
            
            orders_data = api_response.data
            
            for order_data in orders_data:
                order_id = order_data.get('id')
                status = order_data.get('status')
                symbol = order_data.get('symbol')
                
                status_counts[status] += 1
                
                # Handle filled orders
                if status == 'filled':
                    if order_id not in [o.get('id') for o in self.filled_orders]:
                        self.filled_orders.append(order_data)
                        logger.info(f"Order filled: {order_id} - {symbol}")
                
                # Handle rejected orders
                elif status == 'rejected':
                    if order_id not in [o.get('id') for o in self.rejected_orders]:
                        self.rejected_orders.append(order_data)
                        logger.warning(f"Order rejected: {order_id} - {symbol}")
                
                # Update submitted orders tracking
                if order_id in self.submitted_orders:
                    self.submitted_orders[order_id].update(order_data)
            
            return dict(status_counts)
            
        except Exception as e:
            logger.error(f"Exception updating orders: {e}")
            return status_counts
    
    def verify_fill_integrity(self) -> List[str]:
        """Verify order fills against positions (detect API inconsistencies)"""
        discrepancies = []
        
        try:
            # Calculate expected positions from fills
            expected_positions = defaultdict(lambda: {'qty': 0, 'value': 0.0})
            
            for fill in self.filled_orders:
                symbol = fill.get('symbol')
                qty = int(fill.get('filled_qty', 0))
                side = fill.get('side')
                fill_price = float(fill.get('filled_avg_price', 0))
                
                if side == 'buy':
                    expected_positions[symbol]['qty'] += qty
                    expected_positions[symbol]['value'] += qty * fill_price
                elif side == 'sell':
                    expected_positions[symbol]['qty'] -= qty
                    expected_positions[symbol]['value'] -= qty * fill_price
            
            # Compare with actual positions
            for symbol, expected in expected_positions.items():
                expected_qty = expected['qty']
                
                if symbol in self.positions:
                    actual_qty = self.positions[symbol].quantity
                    if self.positions[symbol].side == 'short':
                        actual_qty = -actual_qty
                    
                    if abs(expected_qty - actual_qty) > 0:
                        discrepancy = f"Position mismatch for {symbol}: Expected {expected_qty}, Actual {actual_qty}"
                        discrepancies.append(discrepancy)
                        logger.warning(discrepancy)
                else:
                    if expected_qty != 0:
                        discrepancy = f"Missing position for {symbol}: Expected {expected_qty}, Found 0"
                        discrepancies.append(discrepancy)
                        logger.warning(discrepancy)
            
            return discrepancies
            
        except Exception as e:
            error_msg = f"Error verifying fill integrity: {e}"
            logger.error(error_msg)
            return [error_msg]
    
    def start_monitoring(self, update_interval: int = 30):
        """Start order and position monitoring thread"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        
        def monitor_loop():
            while self.is_monitoring:
                try:
                    # Update positions
                    self.update_positions()
                    
                    # Update orders
                    order_status = self.update_orders()
                    
                    # Verify integrity
                    discrepancies = self.verify_fill_integrity()
                    
                    if discrepancies:
                        logger.warning(f"Found {len(discrepancies)} position discrepancies")
                    
                    # Log status
                    total_orders = sum(order_status.values())
                    if total_orders > 0:
                        logger.info(f"Order status: {dict(order_status)}")
                    
                    time.sleep(update_interval)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(60)  # Wait longer on error
        
        self.order_update_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.order_update_thread.start()
        logger.info("Order monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring thread"""
        self.is_monitoring = False
        if self.order_update_thread:
            self.order_update_thread.join(timeout=5)
        logger.info("Order monitoring stopped")
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""
        total_value = 0
        total_pnl = 0
        positions_summary = []
        
        with self.position_lock:
            for symbol, position in self.positions.items():
                pos_summary = {
                    'symbol': symbol,
                    'quantity': position.quantity,
                    'side': position.side,
                    'avg_price': position.avg_price,
                    'market_price': position.market_price,
                    'market_value': position.market_value,
                    'unrealized_pnl': position.unrealized_pnl,
                    'unrealized_pnl_percent': position.unrealized_pnl_percent
                }
                positions_summary.append(pos_summary)
                total_value += position.market_value
                total_pnl += position.unrealized_pnl
        
        return {
            'total_positions': len(self.positions),
            'total_market_value': total_value,
            'total_unrealized_pnl': total_pnl,
            'total_unrealized_pnl_percent': (total_pnl / total_value * 100) if total_value > 0 else 0,
            'positions': positions_summary,
            'daily_trades': self.daily_trades,
            'daily_volume': self.daily_volume,
            'pending_orders': len(self.pending_orders),
            'filled_orders': len(self.filled_orders),
            'rejected_orders': len(self.rejected_orders),
            'last_update': self.last_position_update
        }


# from rest_api_client import AlpacaRESTClient  # Import from previous section


print("Testing Order Management System...")

# Initialize
api_client = AlpacaRESTClient()
semiconductor_symbols = {'NVDA', 'AMD', 'INTC', 'TSM', 'AVGO'}
order_manager = OrderManager(api_client, semiconductor_symbols)

# Test order validation
print("\\n1. Testing order validation...")

# Valid order (limit far from market)
valid_order = OrderRequest(
    symbol="NVDA",
    quantity=10,
    side="buy",
    order_type="limit",
    limit_price=300.0,  # Assuming current price is ~450, this is far enough
    strategy_id="test_strategy"
)

# Test validation
is_valid, msg = order_manager.validator.validate_order(
    valid_order, current_price=450.0, account_value=50000.0, current_positions={}
)
print(f"Valid order test: {is_valid} - {msg}")

# Invalid order (limit too close to market)
invalid_order = OrderRequest(
    symbol="NVDA",
    quantity=10,
    side="buy",
    order_type="limit",
    limit_price=448.0,  # Too close to market price of 450
    strategy_id="test_strategy"
)

is_valid, msg = order_manager.validator.validate_order(
    invalid_order, current_price=450.0, account_value=50000.0, current_positions={}
)
print(f"Invalid order test: {is_valid} - {msg}")

# Test position updates
print("\\n2. Testing position updates...")
success = order_manager.update_positions()
print(f"Position update: {success}")

# Test order status updates
print("\\n3. Testing order status updates...")
status_counts = order_manager.update_orders()
print(f"Order status counts: {status_counts}")

# Get portfolio summary
print("\\n4. Portfolio summary...")
summary = order_manager.get_portfolio_summary()
print(f"Total positions: {summary['total_positions']}")
print(f"Total market value: ${summary['total_market_value']:,.2f}")
print(f"Daily trades: {summary['daily_trades']}")

# Start monitoring for a short test
print("\\n5. Starting monitoring (5 seconds)...")
order_manager.start_monitoring(update_interval=2)
time.sleep(5)
order_manager.stop_monitoring()
print("Order Management System testing complete.")