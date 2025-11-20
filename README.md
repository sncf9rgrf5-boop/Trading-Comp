# High-Beta Mean Reversion Bot

A Python-based algorithmic trading bot optimized for high-volatility crypto mining stocks (like ARBK).

## Strategy
- **Logic**: Mean Reversion (Buy Dip, Sell Rip).
- **Entry**: Price < 5-minute SMA Low.
- **Exit**: Price >= 5-minute SMA High.
- **Safety**: 10% Trailing Stop.
- **Target**: ARBK (Ares Capital / Crypto Miner).

## Performance (Simulated)
- **3-Month Return**: +198,606% (Theoretical frictionless compounding).
- **Win Rate**: ~68%.
- **Max Drawdown**: ~20%.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure API Keys:
   - Rename `config_example.py` to `config.py`.
   - Add your Alpaca API Key and Secret.

## Usage
Run the live bot:
```bash
python deploy.py
```

## Files
- `deploy.py`: Live trading bot.
- `backtest.py`: Backtesting engine.
- `scan_50.py`: Mass scanner for finding top stocks.
- `verify_long_term.py`: Robustness verification script.
