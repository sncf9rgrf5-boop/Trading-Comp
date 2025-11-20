import backtest
import config
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

def verify_long_term():
    tickers = ['ARBK', 'MIGI', 'PLUG']
    results = []
    
    print("Starting 3-Month Robustness Verification...")
    
    for ticker in tickers:
        print(f"\nTesting {ticker}...")
        
        # Fetch full data (6 months)
        data = backtest.fetch_data(symbol=ticker, random_slice=False)
        
        if data.empty:
            print(f"No data for {ticker}")
            continue
            
        # Slice last 3 months (approx 90 days)
        if len(data) > 0:
            end_date = data.index[-1]
            start_date = end_date - timedelta(days=90)
            data = data[data.index >= start_date].copy()
            
            print(f"Data Period: {data.index[0].date()} to {data.index[-1].date()}")
        else:
            print("Data empty after fetch.")
            continue
        
        # Calculate Indicators
        data = backtest.calculate_indicators(data, window=5)
        
        # Run Backtest (Optimized: No Filters)
        values, trades = backtest.run_backtest(
            data, 
            backtest.strategy_mean_reversion, 
            ticker, 
            strategy_params={'stop_loss_pct': None}
        )
        
        if not values:
            print("No trades generated.")
            continue
            
        # Metrics
        final_val = values[-1]
        total_return = ((final_val - config.START_CAPITAL) / config.START_CAPITAL) * 100
        trade_count = len(trades)
        
        # Win Rate
        wins = 0
        for i in range(0, len(trades) - 1, 2):
            if trades[i]['type'] == 'buy' and trades[i+1]['type'] == 'sell':
                if trades[i+1]['price'] > trades[i]['price']:
                    wins += 1
        win_rate = (wins / (trade_count / 2)) * 100 if trade_count > 0 else 0
        
        # Max Drawdown
        equity_curve = pd.Series(values)
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        results.append({
            'Stock': ticker,
            'Return (%)': total_return,
            'Max Drawdown (%)': max_drawdown,
            'Win Rate (%)': win_rate,
            'Trades': trade_count
        })
        
        # Plot Equity Curve
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, values, label=f'{ticker} (Return: {total_return:.1f}%)')
        plt.title(f'{ticker} - 3 Month Performance')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{ticker}_3month.png')
        print(f"Chart saved to {ticker}_3month.png")

    # Summary
    results_df = pd.DataFrame(results)
    print("\n3-Month Verification Results:")
    print(results_df.to_string(index=False))
    results_df.to_csv('long_term_verification.csv', index=False)

if __name__ == "__main__":
    verify_long_term()
