import alpaca_trade_api as tradeapi
import config
import pandas as pd
import time
import json
import os
from datetime import datetime
import math

# --- Configuration ---
SYMBOL = 'ARBK'
TIMEFRAME = '1Min'
WINDOW = 5
TRAILING_STOP_PCT = 0.10
STATE_FILE = 'bot_state.json'

# --- Alpaca Connection ---
api = tradeapi.REST(config.API_KEY, config.SECRET_KEY, config.BASE_URL, api_version='v2')

def get_account():
    return api.get_account()

def get_position(symbol):
    try:
        return api.get_position(symbol)
    except:
        return None

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {'max_price_since_entry': 0.0}

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)

def fetch_data(symbol):
    # Fetch enough data for rolling window
    # We need at least WINDOW + some buffer. Let's get 20 bars.
    # Note: Alpaca get_bars returns historical data. For live, we want the most recent completed bars.
    # We'll use 'limit' to get the last N bars.
    try:
        bars = api.get_bars(symbol, TIMEFRAME, limit=20).df
        if bars.empty:
            return None
        return bars
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def calculate_indicators(df):
    df['avg_low'] = df['low'].rolling(window=WINDOW).mean()
    df['avg_high'] = df['high'].rolling(window=WINDOW).mean()
    return df

def run_bot():
    print(f"--- Starting Live Bot for {SYMBOL} ---")
    print(f"Strategy: Mean Reversion (Window={WINDOW})")
    print(f"Safety: Trailing Stop {TRAILING_STOP_PCT*100}%")
    
    state = load_state()
    
    while True:
        try:
            # 1. Sync / Wait (Simple sleep for now, ideally sync to minute boundary)
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checking market...")
            
            # 2. Get Data
            df = fetch_data(SYMBOL)
            if df is None or len(df) < WINDOW:
                print("Not enough data yet.")
                time.sleep(60)
                continue
                
            df = calculate_indicators(df)
            current_bar = df.iloc[-1]
            
            # Note: In live trading, the 'current' bar is constantly updating. 
            # Ideally we trade on the CLOSE of the previous bar or real-time price.
            # For this strategy (Mean Reversion), we often want to buy THE DIP as it happens.
            # So we will use the LATEST trade price against the calculated averages from completed bars.
            # However, get_bars usually returns completed bars.
            # Let's use the latest completed bar's averages, and compare with REAL-TIME price.
            
            latest_trade = api.get_latest_trade(SYMBOL)
            current_price = latest_trade.price
            
            avg_low = current_bar['avg_low']
            avg_high = current_bar['avg_high']
            
            print(f"Price: ${current_price:.4f} | Avg Low: ${avg_low:.4f} | Avg High: ${avg_high:.4f}")
            
            # 3. Check Position
            position = get_position(SYMBOL)
            
            if position is None:
                # --- NO POSITION: Look for BUY ---
                # Reset state if we are flat
                if state['max_price_since_entry'] > 0:
                    state['max_price_since_entry'] = 0.0
                    save_state(state)
                
                if current_price < avg_low:
                    print(f"SIGNAL: BUY (Price {current_price} < Avg Low {avg_low})")
                    
                    # Calculate Size (95% of Buying Power)
                    account = get_account()
                    buying_power = float(account.buying_power)
                    target_amt = buying_power * 0.95
                    qty = int(target_amt / current_price)
                    
                    if qty > 0:
                        print(f"Submitting BUY order for {qty} shares...")
                        api.submit_order(
                            symbol=SYMBOL,
                            qty=qty,
                            side='buy',
                            type='market',
                            time_in_force='day'
                        )
                        # Initialize State
                        state['max_price_since_entry'] = current_price
                        save_state(state)
                        print("Order Submitted.")
                    else:
                        print("Insufficient funds to buy.")
            
            else:
                # --- HAVE POSITION: Look for SELL ---
                qty = int(position.qty)
                entry_price = float(position.avg_entry_price)
                
                # Update Trailing Stop State
                if current_price > state['max_price_since_entry']:
                    state['max_price_since_entry'] = current_price
                    save_state(state)
                    print(f"New Max Price: ${current_price:.4f}")
                
                max_price = state['max_price_since_entry']
                stop_price = max_price * (1 - TRAILING_STOP_PCT)
                
                print(f"Position: {qty} shares @ ${entry_price:.4f} | Max: ${max_price:.4f} | Stop: ${stop_price:.4f}")
                
                # Check Conditions
                sell_signal = False
                reason = ""
                
                # 1. Take Profit / Mean Reversion Exit
                if current_price >= avg_high:
                    sell_signal = True
                    reason = f"Target Reached (Price {current_price} >= Avg High {avg_high})"
                    
                # 2. Trailing Stop
                elif current_price < stop_price:
                    sell_signal = True
                    reason = f"Trailing Stop Hit (Price {current_price} < Stop {stop_price})"
                    
                if sell_signal:
                    print(f"SIGNAL: SELL - {reason}")
                    print(f"Submitting SELL order for {qty} shares...")
                    api.submit_order(
                        symbol=SYMBOL,
                        qty=qty,
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )
                    # Reset State
                    state['max_price_since_entry'] = 0.0
                    save_state(state)
                    print("Order Submitted.")
            
            # Sleep for 1 minute (simple loop)
            # For production, use a scheduler or websocket for real-time updates
            print("Waiting for next check...")
            time.sleep(60)
            
        except Exception as e:
            print(f"CRITICAL ERROR: {e}")
            time.sleep(60)

if __name__ == "__main__":
    run_bot()
