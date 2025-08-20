# Performance Analysis & Risk Reporting System
import pandas as pd
import numpy as np
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import threading
import time

logger = logging.getLogger(__name__)

@dataclass
class TradeRecord:
    """Complete trade record for performance analysis"""
    trade_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: int
    fill_price: float
    fill_time: datetime
    strategy_id: str
    order_type: str
    commission: float = 0.0
    slippage: float = 0.0
    market_price_at_order: float = 0.0
    
    @property
    def gross_amount(self) -> float:
        """Gross trade amount"""
        return self.quantity * self.fill_price
    
    @property
    def net_amount(self) -> float:
        """Net trade amount after costs"""
        return self.gross_amount - self.commission - self.slippage

@dataclass
class PositionRecord:
    """Position record for tracking"""
    symbol: str
    quantity: int
    avg_entry_price: float
    current_price: float
    timestamp: datetime
    strategy_id: str
    
    @property
    def market_value(self) -> float:
        return abs(self.quantity) * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        if self.quantity > 0:  # Long position
            return self.quantity * (self.current_price - self.avg_entry_price)
        else:  # Short position
            return abs(self.quantity) * (self.avg_entry_price - self.current_price)

@dataclass
class RiskMetrics:
    """Risk assessment metrics"""
    symbol: str
    var_95: float  # 1-day VaR at 95% confidence
    var_99: float  # 1-day VaR at 99% confidence
    expected_shortfall: float  # Expected loss beyond VaR
    max_drawdown: float
    volatility: float
    beta_to_spy: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    timestamp: datetime

class DatabaseManager:
    """SQLite database manager for trade and performance data"""
    
    def __init__(self, db_path: str = "trading_performance.db"):  
        self.db_path = db_path
        self._connection = None
        self.init_database()
    
    def get_connection(self):
        """Get database connection, reusing for in-memory databases"""
        if self.db_path == ":memory:":
            if self._connection is None:
                self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
            return self._connection
        else:
            return sqlite3.connect(self.db_path)
    
    def init_database(self):
        """Initialize database tables"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Enable foreign keys
            cursor.execute('PRAGMA foreign_keys = ON')

            # Trades table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                fill_price REAL NOT NULL,
                fill_time TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                order_type TEXT NOT NULL,
                commission REAL DEFAULT 0.0,
                slippage REAL DEFAULT 0.0,
                market_price_at_order REAL DEFAULT 0.0,
                gross_amount REAL,
                net_amount REAL,
                UNIQUE(trade_id)
            )''')
            
            # Positions table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                avg_entry_price REAL NOT NULL,
                current_price REAL NOT NULL,
                timestamp TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                market_value REAL,
                unrealized_pnl REAL
            )''')
            
            # Risk metrics table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                var_95 REAL NOT NULL,
                var_99 REAL NOT NULL,
                expected_shortfall REAL NOT NULL,
                max_drawdown REAL NOT NULL,
                volatility REAL NOT NULL,
                beta_to_spy REAL NOT NULL,
                sharpe_ratio REAL NOT NULL,
                sortino_ratio REAL NOT NULL,
                calmar_ratio REAL NOT NULL,
                timestamp TEXT NOT NULL
            )''')
            
            # Portfolio snapshots table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_value REAL NOT NULL,
                cash REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                realized_pnl REAL NOT NULL,
                daily_pnl REAL NOT NULL,
                positions_json TEXT NOT NULL
            )''')
            
            # Create indices for better query performance
            cursor.execute('''CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)''')
            cursor.execute('''CREATE INDEX IF NOT EXISTS idx_trades_time ON trades(fill_time)''')
            cursor.execute('''CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)''')
            cursor.execute('''CREATE INDEX IF NOT EXISTS idx_risk_metrics_symbol ON risk_metrics(symbol)''')
            
            conn.commit()
            
            # Only close if not in-memory database
            if self.db_path != ":memory:":
                conn.close()
                
            logger.info(f"Database initialized successfully: {self.db_path}")
        
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def save_trade(self, trade: TradeRecord):
        """Save trade record to database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT OR REPLACE INTO trades 
        (trade_id, symbol, side, quantity, fill_price, fill_time, strategy_id, 
         order_type, commission, slippage, market_price_at_order, gross_amount, net_amount)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade.trade_id, trade.symbol, trade.side, trade.quantity, trade.fill_price,
            trade.fill_time.isoformat(), trade.strategy_id, trade.order_type,
            trade.commission, trade.slippage, trade.market_price_at_order,
            trade.gross_amount, trade.net_amount
        ))
        
        conn.commit()
        
        # Only close if not in-memory database
        if self.db_path != ":memory:":
            conn.close()
    
    def save_position_snapshot(self, positions: List[PositionRecord]):
        """Save position snapshot"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        for position in positions:
            cursor.execute('''
            INSERT INTO positions 
            (symbol, quantity, avg_entry_price, current_price, timestamp, 
             strategy_id, market_value, unrealized_pnl)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                position.symbol, position.quantity, position.avg_entry_price,
                position.current_price, timestamp, position.strategy_id,
                position.market_value, position.unrealized_pnl
            ))
        
        conn.commit()
        
        # Only close if not in-memory database
        if self.db_path != ":memory:":
            conn.close()
    
    def save_risk_metrics(self, metrics: RiskMetrics):
        """Save risk metrics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO risk_metrics 
        (symbol, var_95, var_99, expected_shortfall, max_drawdown, volatility,
         beta_to_spy, sharpe_ratio, sortino_ratio, calmar_ratio, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.symbol, metrics.var_95, metrics.var_99, metrics.expected_shortfall,
            metrics.max_drawdown, metrics.volatility, metrics.beta_to_spy,
            metrics.sharpe_ratio, metrics.sortino_ratio, metrics.calmar_ratio,
            metrics.timestamp.isoformat()
        ))
        
        conn.commit()
        
        # Only close if not in-memory database
        if self.db_path != ":memory:":
            conn.close()
    
    def save_portfolio_snapshot(self, total_value: float, cash: float, unrealized_pnl: float,
                               realized_pnl: float, daily_pnl: float, positions: List[PositionRecord]):
        """Save complete portfolio snapshot"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        positions_json = json.dumps([asdict(pos) for pos in positions], default=str)
        
        cursor.execute('''
        INSERT INTO portfolio_snapshots 
        (timestamp, total_value, cash, unrealized_pnl, realized_pnl, daily_pnl, positions_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(), total_value, cash, unrealized_pnl,
            realized_pnl, daily_pnl, positions_json
        ))
        
        conn.commit()
        
        # Only close if not in-memory database
        if self.db_path != ":memory:":
            conn.close()
    
    def get_trades(self, symbol: Optional[str] = None, start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None, strategy_id: Optional[str] = None) -> pd.DataFrame:
        """Retrieve trades with optional filters"""
        conn = self.get_connection()
        
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if start_date:
            query += " AND fill_time >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND fill_time <= ?"
            params.append(end_date.isoformat())
        
        if strategy_id:
            query += " AND strategy_id = ?"
            params.append(strategy_id)
        
        query += " ORDER BY fill_time"
        
        df = pd.read_sql_query(query, conn, params=params)
        
        # Only close if not in-memory database
        if self.db_path != ":memory:":
            conn.close()
        
        if not df.empty:
            df['fill_time'] = pd.to_datetime(df['fill_time'])
        
        return df
    
    def get_portfolio_history(self, days: int = 30) -> pd.DataFrame:
        """Get portfolio history for specified days"""
        conn = self.get_connection()
        
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        query = '''
        SELECT * FROM portfolio_snapshots 
        WHERE timestamp >= ? 
        ORDER BY timestamp
        '''
        
        df = pd.read_sql_query(query, conn, params=[start_date])
        
        # Only close if not in-memory database
        if self.db_path != ":memory:":
            conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df

class PerformanceAnalyzer:
    """Comprehensive performance analysis system"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
    
    def calculate_returns_series(self, trades_df: pd.DataFrame) -> pd.Series:
        """Calculate returns series from trades"""
        if trades_df.empty:
            return pd.Series(dtype=float)
        
        # Group by symbol and calculate P&L from trades
        returns_data = []
        
        for symbol in trades_df['symbol'].unique():
            symbol_trades = trades_df[trades_df['symbol'] == symbol].sort_values('fill_time')
            
            position = 0
            avg_price = 0
            
            for _, trade in symbol_trades.iterrows():
                if trade['side'] == 'buy':
                    if position >= 0:
                        # Adding to long position or initiating long
                        total_value = position * avg_price + trade['quantity'] * trade['fill_price']
                        total_quantity = position + trade['quantity']
                        avg_price = total_value / total_quantity if total_quantity > 0 else 0
                        position = total_quantity
                    else:
                        # Covering short position
                        cover_qty = min(trade['quantity'], abs(position))
                        pnl = cover_qty * (avg_price - trade['fill_price'])
                        returns_data.append({
                            'timestamp': trade['fill_time'],
                            'pnl': pnl - trade['commission'] - trade['slippage'],
                            'symbol': symbol
                        })
                        position += cover_qty
                        
                        if trade['quantity'] > cover_qty:
                            # Remaining quantity goes long
                            remaining = trade['quantity'] - cover_qty
                            position = remaining
                            avg_price = trade['fill_price']
                
                elif trade['side'] == 'sell':
                    if position > 0:
                        # Reducing long position
                        sell_qty = min(trade['quantity'], position)
                        pnl = sell_qty * (trade['fill_price'] - avg_price)
                        returns_data.append({
                            'timestamp': trade['fill_time'],
                            'pnl': pnl - trade['commission'] - trade['slippage'],
                            'symbol': symbol
                        })
                        position -= sell_qty
                        
                        if trade['quantity'] > sell_qty:
                            # Remaining quantity goes short
                            remaining = trade['quantity'] - sell_qty
                            position = -remaining
                            avg_price = trade['fill_price']
                    else:
                        # Adding to short position or initiating short
                        total_value = abs(position) * avg_price + trade['quantity'] * trade['fill_price']
                        total_quantity = abs(position) + trade['quantity']
                        avg_price = total_value / total_quantity if total_quantity > 0 else 0
                        position = -total_quantity
        
        if not returns_data:
            return pd.Series(dtype=float)
        
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.set_index('timestamp').sort_index()
        
        # Resample to daily returns
        daily_returns = returns_df.groupby(returns_df.index.date)['pnl'].sum()
        return pd.Series(daily_returns.values, index=pd.to_datetime(daily_returns.index))
    
    def calculate_performance_metrics(self, returns: pd.Series, 
                                    initial_capital: float = 50000) -> Dict:
        """Calculate comprehensive performance metrics"""
        if returns.empty:
            return self._empty_metrics()
        
        # Basic statistics
        total_return = returns.sum()
        returns_pct = returns / initial_capital
        
        # Annualized metrics
        trading_days = len(returns)
        mean_daily_return = returns_pct.mean()
        annual_return = mean_daily_return * 252
        
        volatility = returns_pct.std() * np.sqrt(252)
        
        # Risk-adjusted returns
        excess_returns = returns_pct - (self.risk_free_rate / 252)
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
        
        # Downside deviation (for Sortino ratio)
        downside_returns = returns_pct[returns_pct < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Drawdown analysis
        cumulative_returns = (1 + returns_pct).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        # Profit factor
        wins = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        profit_factor = wins / losses if losses > 0 else float('inf')
        
        # Additional metrics
        avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
        avg_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0
        
        return {
            'total_return': total_return,
            'total_return_percent': (total_return / initial_capital) * 100,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'current_drawdown': drawdown.iloc[-1] if len(drawdown) > 0 else 0,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(returns),
            'trading_days': trading_days,
            'avg_daily_return': mean_daily_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'best_day': returns.max(),
            'worst_day': returns.min(),
            'recovery_factor': abs(total_return / max_drawdown) if max_drawdown != 0 else 0
        }
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics for no trades"""
        return {
            'total_return': 0, 'total_return_percent': 0, 'annual_return': 0,
            'volatility': 0, 'sharpe_ratio': 0, 'sortino_ratio': 0, 'calmar_ratio': 0,
            'max_drawdown': 0, 'current_drawdown': 0, 'win_rate': 0, 'profit_factor': 0,
            'total_trades': 0, 'trading_days': 0, 'avg_daily_return': 0,
            'avg_win': 0, 'avg_loss': 0, 'best_day': 0, 'worst_day': 0, 'recovery_factor': 0
        }
    
    def calculate_var_metrics(self, returns: pd.Series, confidence_levels: List[float] = [0.95, 0.99]) -> Dict:
        """Calculate Value at Risk metrics"""
        if returns.empty:
            return {f'var_{int(c*100)}': 0 for c in confidence_levels}
        
        var_metrics = {}
        
        for confidence in confidence_levels:
            var_value = np.percentile(returns, (1 - confidence) * 100)
            var_metrics[f'var_{int(confidence*100)}'] = var_value
            
            # Expected Shortfall (Conditional VaR)
            tail_losses = returns[returns <= var_value]
            expected_shortfall = tail_losses.mean() if len(tail_losses) > 0 else 0
            var_metrics[f'es_{int(confidence*100)}'] = expected_shortfall
        
        return var_metrics
    
    def calculate_trading_costs(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate trading cost analysis"""
        if trades_df.empty:
            return {
                'total_commission': 0, 'total_slippage': 0, 'total_costs': 0,
                'cost_per_trade': 0, 'cost_percentage': 0, 'turnover': 0
            }
        
        total_commission = trades_df['commission'].sum()
        total_slippage = trades_df['slippage'].sum()
        total_costs = total_commission + total_slippage
        
        total_volume = trades_df['gross_amount'].sum()
        cost_percentage = (total_costs / total_volume) * 100 if total_volume > 0 else 0
        
        return {
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'total_costs': total_costs,
            'cost_per_trade': total_costs / len(trades_df),
            'cost_percentage': cost_percentage,
            'total_volume': total_volume,
            'turnover': total_volume / 50000  # Assuming 50k initial capital
        }
    
    def analyze_strategy_performance(self, start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None) -> Dict:
        """Analyze performance by strategy"""
        trades_df = self.db_manager.get_trades(start_date=start_date, end_date=end_date)
        
        if trades_df.empty:
            return {}
        
        strategy_performance = {}
        
        for strategy_id in trades_df['strategy_id'].unique():
            strategy_trades = trades_df[trades_df['strategy_id'] == strategy_id]
            returns = self.calculate_returns_series(strategy_trades)
            performance = self.calculate_performance_metrics(returns)
            cost_analysis = self.calculate_trading_costs(strategy_trades)
            
            strategy_performance[strategy_id] = {
                'performance': performance,
                'costs': cost_analysis,
                'trade_count': len(strategy_trades)
            }
        
        return strategy_performance

class RiskDashboard:
    """Risk monitoring and reporting dashboard"""
    
    def __init__(self, db_manager: DatabaseManager, performance_analyzer: PerformanceAnalyzer):
        self.db_manager = db_manager
        self.performance_analyzer = performance_analyzer
        self.monitoring_thread = None
        self.is_monitoring = False
        
        # Risk thresholds for semiconductor trading
        self.max_daily_var = 1500  # $1500 daily VaR limit
        self.max_portfolio_drawdown = 0.12  # 12% max drawdown
        self.min_sharpe_ratio = 0.4  # Minimum acceptable Sharpe ratio
        self.max_concentration = 0.25  # 25% max per symbol
    
    def calculate_real_time_risk(self, positions: List[PositionRecord]) -> Dict:
        """Calculate real-time risk metrics"""
        if not positions:
            return self._empty_risk_metrics()
        
        # Portfolio value and concentration
        total_value = sum(pos.market_value for pos in positions)
        position_weights = {pos.symbol: pos.market_value / total_value for pos in positions if total_value > 0}
        
        # Concentration risk
        max_concentration = max(position_weights.values()) if position_weights else 0
        herfindahl_index = sum(w**2 for w in position_weights.values())
        
        # Calculate portfolio volatility (simplified using semiconductor correlations)
        # Semiconductor stocks typically have correlation of 0.6-0.8
        avg_correlation = 0.7
        position_volatilities = []
        
        for pos in positions:
            # Estimate individual stock volatility based on semiconductor characteristics
            if pos.symbol == 'NVDA':
                estimated_vol = 0.035  # 3.5% daily vol for NVDA
            elif pos.symbol in ['AMD', 'AVGO']:
                estimated_vol = 0.030  # 3% daily vol
            else:  # INTC, TSM
                estimated_vol = 0.025  # 2.5% daily vol
            
            position_volatilities.append(estimated_vol * pos.market_value)
        
        # Portfolio volatility with correlation adjustment
        if len(position_volatilities) > 1:
            portfolio_variance = sum(v**2 for v in position_volatilities)
            # Add covariance terms
            for i in range(len(position_volatilities)):
                for j in range(i+1, len(position_volatilities)):
                    portfolio_variance += 2 * avg_correlation * position_volatilities[i] * position_volatilities[j]
            portfolio_volatility = np.sqrt(portfolio_variance)
        else:
            portfolio_volatility = sum(position_volatilities)
        
        # VaR calculation (parametric approach)
        var_95 = portfolio_volatility * 1.645  # 95% confidence
        var_99 = portfolio_volatility * 2.326  # 99% confidence
        
        return {
            'total_portfolio_value': total_value,
            'max_concentration': max_concentration,
            'herfindahl_index': herfindahl_index,
            'portfolio_volatility': portfolio_volatility,
            'var_95': var_95,
            'var_99': var_99,
            'positions_count': len(positions),
            'position_weights': position_weights,
            'timestamp': datetime.now()
        }
    
    def _empty_risk_metrics(self) -> Dict:
        """Return empty risk metrics"""
        return {
            'total_portfolio_value': 0, 'max_concentration': 0, 'herfindahl_index': 0,
            'portfolio_volatility': 0, 'var_95': 0, 'var_99': 0, 'positions_count': 0,
            'position_weights': {}, 'timestamp': datetime.now()
        }
    
    def check_risk_limits(self, risk_metrics: Dict) -> List[str]:
        """Check risk limits and return violations"""
        violations = []
        
        if risk_metrics['var_95'] > self.max_daily_var:
            violations.append(f"Daily VaR ${risk_metrics['var_95']:.2f} exceeds limit ${self.max_daily_var}")
        
        if risk_metrics['max_concentration'] > self.max_concentration:
            violations.append(f"Position concentration {risk_metrics['max_concentration']:.1%} exceeds {self.max_concentration:.1%} limit")
        
        # Check recent performance for drawdown
        recent_trades = self.db_manager.get_trades(
            start_date=datetime.now() - timedelta(days=30)
        )
        
        if not recent_trades.empty:
            returns = self.performance_analyzer.calculate_returns_series(recent_trades)
            if not returns.empty:
                performance = self.performance_analyzer.calculate_performance_metrics(returns)
                if performance['current_drawdown'] < -self.max_portfolio_drawdown:
                    violations.append(
                        f"Current drawdown {performance['current_drawdown']:.1%} exceeds "
                        f"limit {self.max_portfolio_drawdown:.1%}"
                    )
        
        return violations
    
    def generate_daily_report(self) -> Dict:
        """Generate comprehensive daily risk report"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        
        # Get today's trades
        daily_trades = self.db_manager.get_trades(start_date=start_date, end_date=end_date)
        
        # Calculate performance
        returns = self.performance_analyzer.calculate_returns_series(daily_trades)
        performance = self.performance_analyzer.calculate_performance_metrics(returns)
        var_metrics = self.performance_analyzer.calculate_var_metrics(returns)
        cost_analysis = self.performance_analyzer.calculate_trading_costs(daily_trades)
        
        # Get monthly performance for comparison
        month_start = end_date - timedelta(days=30)
        monthly_trades = self.db_manager.get_trades(start_date=month_start, end_date=end_date)
        monthly_returns = self.performance_analyzer.calculate_returns_series(monthly_trades)
        monthly_performance = self.performance_analyzer.calculate_performance_metrics(monthly_returns)
        
        # Strategy breakdown
        strategy_performance = self.performance_analyzer.analyze_strategy_performance(
            start_date=start_date, end_date=end_date
        )
        
        return {
            'report_date': end_date.strftime('%Y-%m-%d'),
            'daily_performance': performance,
            'daily_var_metrics': var_metrics,
            'daily_cost_analysis': cost_analysis,
            'monthly_performance': monthly_performance,
            'strategy_performance': strategy_performance,
            'daily_trades_count': len(daily_trades),
            'monthly_trades_count': len(monthly_trades)
        }
    
    def print_risk_dashboard(self, positions: List[PositionRecord]):
        """Print formatted risk dashboard"""
        risk_metrics = self.calculate_real_time_risk(positions)
        violations = self.check_risk_limits(risk_metrics)
        daily_report = self.generate_daily_report()
        
        print(f"\n{'='*80}")
        print(f"SEMICONDUCTOR TRADING RISK DASHBOARD - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        # Portfolio overview
        print(f"\nPORTFOLIO OVERVIEW:")
        print(f"  Total Value: ${risk_metrics['total_portfolio_value']:,.2f}")
        print(f"  Positions: {risk_metrics['positions_count']}")
        print(f"  Max Concentration: {risk_metrics['max_concentration']:.1%}")
        print(f"  Portfolio Volatility: ${risk_metrics['portfolio_volatility']:,.2f}")
        print(f"  Diversification (1/HHI): {1/risk_metrics['herfindahl_index']:.1f}" if risk_metrics['herfindahl_index'] > 0 else "  Diversification: N/A")
        
        # Risk metrics
        print(f"\\nRISK METRICS:")
        print(f"  1-Day VaR (95%): ${risk_metrics['var_95']:,.2f}")
        print(f"  1-Day VaR (99%): ${risk_metrics['var_99']:,.2f}")
        print(f"  Risk Limit Utilization: {(risk_metrics['var_95']/self.max_daily_var)*100:.1f}%")
        
        # Position weights
        if risk_metrics['position_weights']:
            print(f"\\nPOSITION WEIGHTS:")
            for symbol, weight in sorted(risk_metrics['position_weights'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {symbol}: {weight:.1%}")
        
        # Risk violations
        if violations:
            print(f"\\nRISK VIOLATIONS:")
            for violation in violations:
                print(f"  - {violation}")
        else:
            print(f"\\nNo risk limit violations")
        
        # Daily performance
        daily_perf = daily_report['daily_performance']
        print(f"\\nDAILY PERFORMANCE:")
        print(f"  P&L: ${daily_perf['total_return']:,.2f}")
        print(f"  Return: {daily_perf['total_return_percent']:.2f}%")
        print(f"  Trades: {daily_perf['total_trades']}")
        print(f"  Win Rate: {daily_perf['win_rate']:.1%}")
        if daily_perf['total_trades'] > 0:
            print(f"  Avg Win: ${daily_perf['avg_win']:,.2f}")
            print(f"  Avg Loss: ${daily_perf['avg_loss']:,.2f}")
        
        # Monthly performance
        monthly_perf = daily_report['monthly_performance']
        print(f"\\nMONTHLY PERFORMANCE:")
        print(f"  Total Return: {monthly_perf['total_return_percent']:.2f}%")
        print(f"  Sharpe Ratio: {monthly_perf['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio: {monthly_perf['sortino_ratio']:.2f}")
        print(f"  Max Drawdown: {monthly_perf['max_drawdown']:.1%}")
        print(f"  Recovery Factor: {monthly_perf['recovery_factor']:.2f}")
        print(f"  Total Trades: {monthly_perf['total_trades']}")
        
        # Strategy breakdown
        if daily_report['strategy_performance']:
            print(f"\\nSTRATEGY BREAKDOWN:")
            for strategy, data in daily_report['strategy_performance'].items():
                perf = data['performance']
                print(f"  {strategy.upper()}:")
                print(f"    Return: {perf['total_return_percent']:.2f}%")
                print(f"    Trades: {data['trade_count']}")
                print(f"    Win Rate: {perf['win_rate']:.1%}")
                if perf['sharpe_ratio'] != 0:
                    print(f"    Sharpe: {perf['sharpe_ratio']:.2f}")
        
        # Cost analysis
        costs = daily_report['daily_cost_analysis']
        print(f"\\nTRADING COSTS:")
        print(f"  Total Costs: ${costs['total_costs']:,.2f}")
        print(f"  Commission: ${costs['total_commission']:,.2f}")
        print(f"  Slippage: ${costs['total_slippage']:,.2f}")
        print(f"  Cost per Trade: ${costs['cost_per_trade']:,.2f}")
        print(f"  Cost %: {costs['cost_percentage']:.3f}%")
        print(f"  Turnover: {costs['turnover']:.1f}x")
        
        print(f"\\n{'='*80}")
    
    def start_monitoring(self, update_interval: int = 300):
        """Start continuous risk monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        
        def monitor_loop():
            while self.is_monitoring:
                try:
                    # Generate and log daily report
                    daily_report = self.generate_daily_report()
                    
                    # Check for risk violations
                    # This would typically check current positions
                    # For now, we'll check recent performance
                    if daily_report['monthly_performance']['current_drawdown'] < -self.max_portfolio_drawdown:
                        logger.critical(f"Portfolio drawdown {daily_report['monthly_performance']['current_drawdown']:.1%} exceeds limit")
                    
                    if daily_report['daily_performance']['total_trades'] > 0:
                        logger.info(f"Daily P&L: ${daily_report['daily_performance']['total_return']:,.2f}")
                    
                    time.sleep(update_interval)
                    
                except Exception as e:
                    logger.error(f"Error in risk monitoring: {e}")
                    time.sleep(60)
        
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Risk monitoring started")
    
    def stop_monitoring(self):
        """Stop risk monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Risk monitoring stopped")
    
    def export_performance_report(self, filepath: str, days: int = 30):
        """Export comprehensive performance report to file"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get data
        trades_df = self.db_manager.get_trades(start_date=start_date, end_date=end_date)
        returns = self.performance_analyzer.calculate_returns_series(trades_df)
        performance = self.performance_analyzer.calculate_performance_metrics(returns)
        var_metrics = self.performance_analyzer.calculate_var_metrics(returns)
        cost_analysis = self.performance_analyzer.calculate_trading_costs(trades_df)
        strategy_performance = self.performance_analyzer.analyze_strategy_performance(start_date, end_date)
        
        # Create comprehensive report
        report = {
            'report_period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            'generated_at': datetime.now().isoformat(),
            'performance_metrics': performance,
            'var_metrics': var_metrics,
            'cost_analysis': cost_analysis,
            'strategy_breakdown': strategy_performance,
            'trade_count': len(trades_df),
            'symbols_traded': trades_df['symbol'].unique().tolist() if not trades_df.empty else []
        }
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance report exported to {filepath}")
        return report

# Test the Performance Analysis System

print("Testing Performance Analysis & Risk Reporting System...")

# Initialize components
db_manager = DatabaseManager(":memory:")  # In-memory database for testing
performance_analyzer = PerformanceAnalyzer(db_manager)
risk_dashboard = RiskDashboard(db_manager, performance_analyzer)

# Create sample trade data
print("\\n1. Creating sample trade data...")
sample_trades = [
  TradeRecord(
      trade_id="T001",
      symbol="NVDA",
      side="buy",
      quantity=100,
      fill_price=450.0,
      fill_time=datetime.now() - timedelta(days=5),
      strategy_id="trend_following",
      order_type="market",
      commission=1.0,
      slippage=0.5
  ),
  TradeRecord(
      trade_id="T002",
      symbol="NVDA",
      side="sell",
      quantity=100,
      fill_price=455.0,
      fill_time=datetime.now() - timedelta(days=3),
      strategy_id="trend_following",
      order_type="market",
      commission=1.0,
      slippage=0.5
  ),
  TradeRecord(
      trade_id="T003",
      symbol="AMD",
      side="buy",
      quantity=200,
      fill_price=120.0,
      fill_time=datetime.now() - timedelta(days=4),
      strategy_id="mean_reversion",
      order_type="limit",
      commission=1.0,
      slippage=0.3
  ),
  TradeRecord(
      trade_id="T004",
      symbol="AMD",
      side="sell",
      quantity=200,
      fill_price=118.0,
      fill_time=datetime.now() - timedelta(days=2),
      strategy_id="mean_reversion",
      order_type="market",
      commission=1.0,
      slippage=0.4
  )
]

# Save trades
for trade in sample_trades:
  db_manager.save_trade(trade)
print("Sample trades saved to database")

# Test performance calculation
print("\\n2. Testing performance calculation...")
trades_df = db_manager.get_trades()
returns = performance_analyzer.calculate_returns_series(trades_df)
performance = performance_analyzer.calculate_performance_metrics(returns)

print(f"Total return: ${performance['total_return']:.2f}")
print(f"Total trades: {performance['total_trades']}")
print(f"Win rate: {performance['win_rate']:.1%}")
print(f"Sharpe ratio: {performance['sharpe_ratio']:.2f}")

# Test VaR calculation
print("\\n3. Testing VaR calculation...")
var_metrics = performance_analyzer.calculate_var_metrics(returns)
for metric, value in var_metrics.items():
  print(f"{metric}: ${value:.2f}")

# Test cost analysis
print("\\n4. Testing cost analysis...")
cost_analysis = performance_analyzer.calculate_trading_costs(trades_df)
print(f"Total costs: ${cost_analysis['total_costs']:.2f}")
print(f"Cost percentage: {cost_analysis['cost_percentage']:.3f}%")
print(f"Cost per trade: ${cost_analysis['cost_per_trade']:.2f}")

# Test strategy analysis
print("\\n5. Testing strategy analysis...")
strategy_performance = performance_analyzer.analyze_strategy_performance()
for strategy, data in strategy_performance.items():
  perf = data['performance']
  print(f"{strategy}: Return {perf['total_return_percent']:.2f}%, Trades {data['trade_count']}")

# Test risk dashboard
print("\\n6. Testing risk dashboard...")
sample_positions = [
  PositionRecord(
      symbol="NVDA",
      quantity=0,  # Flat position after trades
      avg_entry_price=450.0,
      current_price=455.0,
      timestamp=datetime.now(),
      strategy_id="trend_following"
  ),
  PositionRecord(
      symbol="AMD",
      quantity=0,  # Flat position after trades
      avg_entry_price=120.0,
      current_price=118.0,
      timestamp=datetime.now(),
      strategy_id="mean_reversion"
  )
]

risk_dashboard.print_risk_dashboard(sample_positions)

# Test report export
print("\\n7. Testing report export...")
report = risk_dashboard.export_performance_report("test_report.json", days=7)
print(f"Report exported with {len(report['symbols_traded'])} symbols traded")

print("\\nPerformance Analysis & Risk Reporting System testing complete.")