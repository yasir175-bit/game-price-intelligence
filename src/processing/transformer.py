import pandas as pd
from typing import List
import numpy as np

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame of price history for a game, compute engineered features.
    Required Columns in df: 'timestamp', 'initial_price', 'final_price', 'discount_percent', 'lowest_price_ever'
    """
    if df.empty or len(df) < 2:
        return df

    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Target variable: Will the price drop in the next period? (For future classification model training)
    df['price_drop_indicator'] = (df['final_price'].shift(-1) < df['final_price']).astype(int)
    
    # Feature: Days since last discount
    # Identify discounts
    df['is_discounted'] = (df['discount_percent'] > 0).astype(int)
    
    # Price Drop % relative to initial
    df['price_drop_pct'] = np.where(df['initial_price'] > 0, 
                                    (df['initial_price'] - df['final_price']) / df['initial_price'] * 100, 
                                    0)

    # Rolling Average Price (window of 3 periods, you can change based on frequency)
    df['rolling_avg_price_3'] = df['final_price'].rolling(window=3, min_periods=1).mean()
    
    # Is it at the all time low?
    df['is_current_all_time_low'] = (df['final_price'] <= df['lowest_price_ever']).astype(int)

    return df
