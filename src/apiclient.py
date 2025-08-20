# Core REST API Client with Robust Error Handling
import requests
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import base64
from dataclasses import dataclass
from enum import Enum
import threading
import queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    """Order status enumeration"""
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    DONE_FOR_DAY = "done_for_day"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REPLACED = "replaced"
    PENDING_CANCEL = "pending_cancel"
    PENDING_REPLACE = "pending_replace"
    ACCEPTED = "accepted"
    PENDING_NEW = "pending_new"
    ACCEPTED_FOR_BIDDING = "accepted_for_bidding"
    STOPPED = "stopped"
    REJECTED = "rejected"
    SUSPENDED = "suspended"
    CALCULATED = "calculated"

class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class TimeInForce(Enum):
    """Time in force enumeration"""
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"

@dataclass
class APIResponse:
    """Standardized API response wrapper"""
    success: bool
    data: Optional[Dict] = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    response_time: Optional[float] = None

class AlpacaRESTClient:
    """Enhanced Alpaca REST API client with robust error handling"""
    
    def __init__(self):
        # Alpaca Broker API Configuration
        self.api_key = "CKWWVBPP75ZM7BGWT7C7"
        self.secret_key = "8TqSb4XyuTodeFfZU6pSeDIBfMnU88R9oNiGep1E"
        self.account_id = "2e363eac-3981-351a-afbb-0322a4540912"
        self.account_number = "9144741SW"
        self.base_url = "https://broker-api.sandbox.alpaca.markets"
        
        # Setup authentication
        self.headers = {
            'Authorization': f'Basic {self._encode_credentials()}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        # Connection management
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        self.request_lock = threading.Lock()
        
        # Error tracking
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        self.error_backoff_time = 1.0
        
        # Connection health
        self.is_connected = False
        self.last_successful_request = None
        
    def _encode_credentials(self) -> str:
        """Encode API credentials for Basic Auth"""
        credentials = f"{self.api_key}:{self.secret_key}"
        return base64.b64encode(credentials.encode()).decode('ascii')
    
    def _rate_limit(self):
        """Implement rate limiting to avoid API limits"""
        with self.request_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()
    
    def _handle_error(self, response: requests.Response, endpoint: str) -> APIResponse:
        """Centralized error handling"""
        self.consecutive_errors += 1
        
        error_msg = f"API Error on {endpoint}: {response.status_code}"
        
        try:
            error_data = response.json()
            if 'message' in error_data:
                error_msg += f" - {error_data['message']}"
        except:
            error_msg += f" - {response.text}"
        
        logger.error(error_msg)
        
        # Implement exponential backoff for consecutive errors
        if self.consecutive_errors >= self.max_consecutive_errors:
            backoff_time = self.error_backoff_time * (2 ** (self.consecutive_errors - self.max_consecutive_errors))
            logger.warning(f"Too many consecutive errors. Backing off for {backoff_time:.2f} seconds")
            time.sleep(min(backoff_time, 60))  # Cap at 60 seconds
        
        return APIResponse(
            success=False,
            error=error_msg,
            status_code=response.status_code
        )
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None, 
                     params: Optional[Dict] = None) -> APIResponse:
        """Make HTTP request with comprehensive error handling"""
        self._rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url, params=params, timeout=30)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data, params=params, timeout=30)
            elif method.upper() == 'PUT':
                response = self.session.put(url, json=data, params=params, timeout=30)
            elif method.upper() == 'DELETE':
                response = self.session.delete(url, params=params, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response_time = time.time() - start_time
            
            # Check for success
            if response.status_code in [200, 201, 202]:
                self.consecutive_errors = 0  # Reset error counter on success
                self.is_connected = True
                self.last_successful_request = datetime.now()
                
                try:
                    response_data = response.json()
                except:
                    response_data = {"status": "success"}
                
                return APIResponse(
                    success=True,
                    data=response_data,
                    status_code=response.status_code,
                    response_time=response_time
                )
            else:
                return self._handle_error(response, endpoint)
                
        except requests.exceptions.Timeout:
            logger.error(f"Timeout on {endpoint}")
            return APIResponse(success=False, error="Request timeout")
        
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error on {endpoint}")
            self.is_connected = False
            return APIResponse(success=False, error="Connection error")
        
        except Exception as e:
            logger.error(f"Unexpected error on {endpoint}: {str(e)}")
            return APIResponse(success=False, error=str(e))
    
    def get_account_info(self) -> APIResponse:
        """Get account information with error handling"""
        return self._make_request('GET', f'/v1/accounts/{self.account_id}')
    
    def get_positions(self) -> APIResponse:
        """Get current positions"""
        return self._make_request('GET', f'/v1/trading/accounts/{self.account_id}/positions')
    
    def get_orders(self, status: Optional[str] = None, limit: int = 100) -> APIResponse:
        """Get orders with optional status filter"""
        params = {'limit': limit}
        if status:
            params['status'] = status
        
        return self._make_request('GET', f'/v1/trading/accounts/{self.account_id}/orders', params=params)
    
    def get_order(self, order_id: str) -> APIResponse:
        """Get specific order by ID"""
        return self._make_request('GET', f'/v1/trading/accounts/{self.account_id}/orders/{order_id}')
    
    def place_order(self, symbol: str, qty: int, side: str, order_type: str = "market",
                   time_in_force: str = "day", limit_price: Optional[float] = None,
                   stop_price: Optional[float] = None, trail_price: Optional[float] = None,
                   trail_percent: Optional[float] = None) -> APIResponse:
        """Place order with comprehensive validation"""
        
        # Validate inputs
        if side not in ['buy', 'sell']:
            return APIResponse(success=False, error="Invalid side. Must be 'buy' or 'sell'")
        
        if qty <= 0:
            return APIResponse(success=False, error="Quantity must be positive")
        
        order_data = {
            'symbol': symbol.upper(),
            'qty': str(qty),
            'side': side,
            'type': order_type,
            'time_in_force': time_in_force
        }
        
        # Add price parameters based on order type
        if order_type == "limit" and limit_price:
            order_data['limit_price'] = str(limit_price)
        elif order_type == "stop" and stop_price:
            order_data['stop_price'] = str(stop_price)
        elif order_type == "stop_limit" and limit_price and stop_price:
            order_data['limit_price'] = str(limit_price)
            order_data['stop_price'] = str(stop_price)
        elif order_type == "trailing_stop":
            if trail_price:
                order_data['trail_price'] = str(trail_price)
            elif trail_percent:
                order_data['trail_percent'] = str(trail_percent)
        
        return self._make_request('POST', f'/v1/trading/accounts/{self.account_id}/orders', data=order_data)
    
    def cancel_order(self, order_id: str) -> APIResponse:
        """Cancel specific order"""
        return self._make_request('DELETE', f'/v1/trading/accounts/{self.account_id}/orders/{order_id}')
    
    def cancel_all_orders(self) -> APIResponse:
        """Cancel all open orders - EMERGENCY FUNCTION"""
        return self._make_request('DELETE', f'/v1/trading/accounts/{self.account_id}/orders')
    
    def get_portfolio_history(self, period: str = "1D", timeframe: str = "1Min") -> APIResponse:
        """Get portfolio history"""
        params = {
            'period': period,
            'timeframe': timeframe
        }
        return self._make_request('GET', f'/v1/trading/accounts/{self.account_id}/portfolio/history', params=params)
    
    def health_check(self) -> bool:
        """Check API connection health"""
        try:
            response = self.get_account_info()
            return response.success
        except:
            return False

# Test the REST client

print("Testing Alpaca REST API Client...")

client = AlpacaRESTClient()

# Test connection
print("1. Testing connection...")
account_response = client.get_account_info()
if account_response.success:
    print("Connection successful")
    print(f"Account ID: {account_response.data.get('id', 'N/A')}")
else:
    print(f"Connection failed: {account_response.error}")

# Test positions
print("\n2. Testing positions...")
positions_response = client.get_positions()
if positions_response.success:
    positions = positions_response.data
    print(f"Retrieved {len(positions)} positions")
else:
    print(f"Failed to get positions: {positions_response.error}")

# Test orders
print("\n3. Testing orders...")
orders_response = client.get_orders()
if orders_response.success:
    orders = orders_response.data
    print(f" Retrieved {len(orders)} orders")
else:
    print(f" Failed to get orders: {orders_response.error}")

print(f"\nConnection health: {client.health_check()}")
print("REST API Client testing complete.")