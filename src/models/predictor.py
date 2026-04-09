import pandas as pd
from prophet import Prophet
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

logger = logging.getLogger(__name__)

def forecast_price_prophet(df: pd.DataFrame, periods: int = 30) -> pd.DataFrame:
    """
    Given a DataFrame with ['timestamp', 'final_price'], 
    uses Prophet to forecast future prices.
    Returns the forecast DataFrame.
    """
    if len(df) < 2:
        logger.warning("Not enough data to run Prophet forecast.")
        return pd.DataFrame()
        
    prophet_df = df[['timestamp', 'final_price']].copy()
    prophet_df.rename(columns={'timestamp': 'ds', 'final_price': 'y'}, inplace=True)
    
    # We will pad the data slightly if we only have recent data mimicking history
    # For a real implementation, you'd fetch 1+ years from ITAD/SteamDB.
    
    m = Prophet(daily_seasonality=True, yearly_seasonality=True)
    m.fit(prophet_df)
    
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def train_recommendation_model(full_historical_df: pd.DataFrame):
    """
    Trains a classification model to predict 'Best time to buy'.
    In a real scenario, this requires a large dataset of past prices.
    We will use a RandomForest predicting `price_drop_indicator`.
    """
    features = ['discount_percent', 'rolling_avg_price_3', 'is_current_all_time_low', 'price_drop_pct']
    
    # Drop NaNs
    df_clean = full_historical_df.dropna(subset=features + ['price_drop_indicator'])
    
    if len(df_clean) < 10:
        logger.warning("Not enough data to train global recommendation model.")
        return None, None
        
    X = df_clean[features]
    y = df_clean['price_drop_indicator']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_scaled, y)
    
    return clf, scaler

def predict_deal_recommendation(clf, scaler, current_features: pd.DataFrame) -> dict:
    """
    Takes the trained classifier and current features to output a recommendation.
    Outputs {'buy_now': bool, 'probability_of_drop': float}
    """
    if clf is None:
        return {'buy_now': False, 'probability_of_drop': 0.0, 'status': 'Insufficient Data'}
        
    X = current_features[['discount_percent', 'rolling_avg_price_3', 'is_current_all_time_low', 'price_drop_pct']]
    X_scaled = scaler.transform(X)
    
    probs = clf.predict_proba(X_scaled)[0]
    
    # If the probability of a drop is very low, it's a good time to buy.
    prob_drop = probs[1] # Probability class 1 (price will drop)
    
    buy_now = bool(prob_drop < 0.4 or current_features.iloc[0]['is_current_all_time_low'] == 1)
    
    return {
        'buy_now': buy_now,
        'probability_of_future_drop': float(prob_drop),
        'status': 'Model Active'
    }
