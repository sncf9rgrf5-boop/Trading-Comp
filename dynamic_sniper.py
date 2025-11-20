import backtest
import config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

def dynamic_sniper():
    # Universe: Top performers from previous scans
    tickers = [
        'ARBK', 'MIGI', 'WKHS', 'SENS', 'SNDL', 'BNGO', 'PLUG', 'OCGN', 'VLCN', 'CGC',
        'RIOT', 'OPEN', 'CLSK', 'COIN', 'TSLA', 'PLTR', 'MARA', 'MSTR', 'NVDA', 'SOFI'
    ]
    
    print(f"Initializing Sniper Mode for {len(tickers)} stocks...")
    
    # 1. Fetch Data & Synchronize
    data_store = {}
    min_start_date = None
    max_end_date = None
    
    for ticker in tickers:
        print(f"Fetching {ticker}...")
        # Fetch full available data to ensure we have overlap
        df = backtest.fetch_data(symbol=ticker, random_slice=False)
        
        if df.empty:
            print(f"  No data for {ticker}")
            continue
            
        # Calculate Indicators upfront for speed (simulating real-time calculation)
        df = backtest.calculate_indicators(df, window=5)
        data_store[ticker] = df
        
        if min_start_date is None or df.index[0] > min_start_date:
            min_start_date = df.index[0]
        if max_end_date is None or df.index[-1] < max_end_date:
            max_end_date = df.index[-1]
            
    print(f"\nSynchronized Period: {min_start_date} to {max_end_date}")
    
    # Pick a random 1-week slice within the synchronized period for the simulation
    sim_start = min_start_date + timedelta(days=np.random.randint(0, (max_end_date - min_start_date).days - 7))
    sim_end = sim_start + timedelta(days=7)
    print(f"Simulation Week: {sim_start.date()} to {sim_end.date()}")
    
    # Filter data to simulation period
    for t in list(data_store.keys()):
        df = data_store[t]
        data_store[t] = df[(df.index >= sim_start) & (df.index <= sim_end)].copy()
        
    # 2. Simulation Loop
    # Create a master timeline (union of all indices)
    all_indices = sorted(list(set().union(*[df.index for df in data_store.values()])))
    
    cash = config.START_CAPITAL
    positions = {} # {ticker: {'shares': 0, 'entry_price': 0.0}}
    portfolio_values = []
    trades = []
    
    MAX_POSITIONS = 5
    
    print("Running Sniper Simulation...")
    
    for current_time in all_indices:
        # 1. Update Portfolio Value & Check Exits
        current_portfolio_val = cash
        
        # Check Exits for existing positions
        for ticker in list(positions.keys()):
            if ticker not in data_store or current_time not in data_store[ticker].index:
                continue
                
            row = data_store[ticker].loc[current_time]
            pos_data = positions[ticker]
            price = row['close']
            
            # Sell Logic (Mean Reversion: Price >= Avg High)
            # Note: We could use the optimized "No Stop Loss" or add one. Let's use "No Stop Loss" as per optimization.
            if row['high'] >= row['avg_high']:
                revenue = pos_data['shares'] * price
                cash += revenue
                trades.append({'type': 'sell', 'ticker': ticker, 'price': price, 'time': current_time, 'profit': revenue - (pos_data['shares'] * pos_data['entry_price'])})
                del positions[ticker]
            else:
                current_portfolio_val += pos_data['shares'] * price

        # 2. Scan for Entry (Sniper Logic)
        candidates = []
        
        for ticker, df in data_store.items():
            if current_time not in df.index:
                continue
                
            row = df.loc[current_time]
            
            # Check if valid data
            if pd.isna(row['avg_low']):
                continue
                
            # Calculate Dip Score
            # Score = (Avg Low - Low) / Low
            # Positive score means Low is below Avg Low (Dip)
            if row['low'] < row['avg_low']:
                dip_score = (row['avg_low'] - row['low']) / row['low']
                candidates.append({
                    'ticker': ticker,
                    'dip_score': dip_score,
                    'price': row['close'], # Use close as execution price approximation
                    'row': row
                })
                
        # Rank by Dip Score Descending (Deepest Dip first)
        candidates.sort(key=lambda x: x['dip_score'], reverse=True)
        
        # 3. Execute Entries
        # Only buy if we have slots
        if len(positions) < MAX_POSITIONS and candidates:
            best_candidate = candidates[0] # The Sniper Target
            ticker = best_candidate['ticker']
            
            if ticker not in positions:
                price = best_candidate['price']
                
                # Position Sizing: Split remaining cash / remaining slots? 
                # Or fixed size? Let's use: Cash / (Max - Current)
                slots_available = MAX_POSITIONS - len(positions)
                amount_to_invest = cash / slots_available
                
                shares = int(amount_to_invest / price)
                
                if shares > 0:
                    cost = shares * price
                    cash -= cost
                    positions[ticker] = {'shares': shares, 'entry_price': price}
                    trades.append({'type': 'buy', 'ticker': ticker, 'price': price, 'time': current_time, 'score': best_candidate['dip_score']})
                    
        # Record Value
        # Recalculate with new positions
        current_portfolio_val = cash
        for ticker, pos_data in positions.items():
            if ticker in data_store and current_time in data_store[ticker].index:
                current_portfolio_val += pos_data['shares'] * data_store[ticker].loc[current_time]['close']
            # If missing data for this minute, use last known (simplified)
            
        portfolio_values.append({'time': current_time, 'value': current_portfolio_val})

    # 3. Results
    final_val = portfolio_values[-1]['value']
    total_return = ((final_val - config.START_CAPITAL) / config.START_CAPITAL) * 100
    
    print(f"\nSniper Mode Results:")
    print(f"Final Value: ${final_val:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Total Trades: {len(trades)}")
    
    # Trade Analysis
    wins = 0
    completed_trades = 0
    # Match buys and sells (simplified)
    # Actually we stored profit in sell trade
    total_profit = 0
    for t in trades:
        if t['type'] == 'sell':
            completed_trades += 1
            total_profit += t['profit']
            if t['profit'] > 0:
                wins += 1
                
    win_rate = (wins / completed_trades * 100) if completed_trades > 0 else 0
    print(f"Win Rate: {win_rate:.2f}%")
    
    # Plot
    df_res = pd.DataFrame(portfolio_values)
    plt.figure(figsize=(12, 6))
    plt.plot(df_res['time'], df_res['value'])
    plt.title(f'Sniper Mode Performance (Return: {total_return:.1f}%)')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.savefig('sniper_mode.png')
    print("Chart saved to sniper_mode.png")

if __name__ == "__main__":
    dynamic_sniper()
