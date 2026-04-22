import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Path where the trained ML model will be saved
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "price_predictor_model.pkl")

def train_and_save_model():
    """
    Simulates a historical dataset, trains a legitimate Linear Regression model,
    evaluates its accuracy, and saves it to disk as a serialized .pkl file.
    """
    print("Initializing ML Training Pipeline...")
    np.random.seed(42)
    n_samples = 2000
    
    # 1. Generate historical simulation data
    base_prices = np.random.uniform(500, 5000, n_samples)
    discount_probs = np.random.rand(n_samples)
    discounts = np.zeros(n_samples)
    
    # Assume 40% of games get a random discount between 10% and 90%
    discounts[discount_probs > 0.6] = np.random.uniform(10, 90, np.sum(discount_probs > 0.6))
    
    # Calculate current price given base and discount
    current_prices = base_prices * (1 - (discounts / 100))
    
    # Define Target: Future Price
    # - If discounted: price reverts back to ~90% to 100% of base price
    # - If not discounted: price decays slightly by ~2% to 10%
    future_prices = np.where(
        discounts > 0,
        base_prices * np.random.uniform(0.9, 1.0, n_samples),
        current_prices * np.random.uniform(0.9, 0.98, n_samples)
    )
    
    df = pd.DataFrame({
        'current_price': current_prices,
        'discount_percent': discounts,
        'future_price': future_prices
    })
    
    # 2. Machine Learning pipeline
    X = df[['current_price', 'discount_percent']]
    y = df['future_price']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and fit Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate model locally
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model Trained! MAE: INR {mae:.2f}, R2 Score: {r2:.4f}")
    
    # 3. Serialize and save the trained model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"Model successfully saved to {MODEL_FILE}")
    return model

def predict_future_price(current_price: float, discount_percent: float) -> float:
    """
    Production inference function.
    Loads the trained serialized model from disk (or triggers training if missing)
    and predicts the target variable based on input features.
    """
    # Auto-trigger ML training if the model file does not exist yet.
    if not os.path.exists(MODEL_FILE):
        train_and_save_model()
        
    # Load real trained model
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
        
    # Standardize input for pandas to match training features
    X_input = pd.DataFrame({
        'current_price': [float(current_price)],
        'discount_percent': [float(discount_percent)]
    })
    
    # Perform prediction
    prediction = model.predict(X_input)[0]
    
    return max(0, prediction) # Safety check for non-negative values

# If run directly via terminal, execute training explicitly
if __name__ == "__main__":
    train_and_save_model()
