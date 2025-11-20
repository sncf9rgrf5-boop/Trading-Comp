import backtest
import config
import pandas as pd
import time

def scan_50():
    tickers = [
        'CORZ', 'IREN', 'CIFR', 'SDIG', 'BTBT', 'HIVE', 'ARBK', 'GREE', 'ANY', 'MIGI',
        'GOEV', 'MULN', 'WKHS', 'HYLN', 'VLCN',
        'NVAX', 'OCGN', 'BNGO', 'SENS', 'SAVA',
        'TLRY', 'CGC', 'SNDL', 'ACB', 'CRON',
        'FCEL', 'BE', 'BLDP', 'PLUG',
        'GME', 'AMC', 'BB', 'KOSS', 'EXPR',
        'SOUN', 'BBAI', 'AI', 'IONQ', 'RGTI', 'QUBT',
        'RIG', 'SWN', 'MRO', 'TELL', 'NEXT', 'ET', 'PBR', 'VALE', 'NU', 'GRAB'
    ]
    
    results = []
    print(f"Starting Mass Scan for {len(tickers)} stocks...")
    
    for i, ticker in enumerate(tickers):
        try:
            print(f"[{i+1}/{len(tickers)}] Testing {ticker}...")
            
            # Fetch data (Random Week)
            data = backtest.fetch_data(symbol=ticker)
            
            if data.empty:
                print(f"  No data for {ticker}")
                continue
                
            # Calculate indicators (Optimized Window = 5)
            data = backtest.calculate_indicators(data, window=5)
            
            # Run Backtest (Optimized: No Stop Loss)
            values, trades = backtest.run_backtest(
                data, 
                backtest.strategy_mean_reversion, 
                ticker, 
                strategy_params={'stop_loss_pct': None}
            )
            
            if not values:
                continue
                
            final_val = values[-1]
            total_return = ((final_val - config.START_CAPITAL) / config.START_CAPITAL) * 100
            trade_count = len(trades)
            
            # Win Rate
            wins = 0
            for j in range(0, len(trades) - 1, 2):
                if trades[j]['type'] == 'buy' and trades[j+1]['type'] == 'sell':
                    if trades[j+1]['price'] > trades[j]['price']:
                        wins += 1
            win_rate = (wins / (trade_count / 2)) * 100 if trade_count > 0 else 0
            
            results.append({
                'Stock': ticker,
                'Return (%)': total_return,
                'Final Value': final_val,
                'Trades': trade_count,
                'Win Rate (%)': win_rate
            })
            
            # Sleep to be nice to API
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  Error processing {ticker}: {e}")
            
    # Sort and Report Top 10
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(by='Return (%)', ascending=False)
        
        print("\nTop 10 High-Beta Performers:")
        print(results_df.head(10).to_string(index=False))
        
        results_df.to_csv('mass_scan_results.csv', index=False)
        print("\nFull results saved to mass_scan_results.csv")
    else:
        print("No results generated.")

if __name__ == "__main__":
    scan_50()
