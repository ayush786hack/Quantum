import os
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from data_preprocessing import load_and_preprocess_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "prediction", "fuel_predictor_model.joblib")

def train_and_save_model():
    print("[Model Training] Loading voyage dataset and generating features...")
    X, y, feature_cols = load_and_preprocess_data()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("[Model Training] Training Gradient Boosting Regressor model...")
    model = GradientBoostingRegressor(n_estimators=150, max_depth=6, learning_rate=0.08, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"[Model Training] Evaluation Results on Test Set:")
    print(f"  - R² Score: {r2:.4f}")
    print(f"  - RMSE:     {rmse:.4f} tonnes fuel")
    
    saved_data = {
        "model": model,
        "feature_cols": feature_cols,
        "r2_score": float(r2),
        "rmse": float(rmse)
    }
    
    joblib.dump(saved_data, MODEL_SAVE_PATH)
    print(f"[Model Training] Model successfully saved to {MODEL_SAVE_PATH}")
    return saved_data

if __name__ == "__main__":
    train_and_save_model()
