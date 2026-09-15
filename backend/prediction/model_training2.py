import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from data_preprocessing2 import load_and_preprocess_data


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_SAVE_PATH = os.path.join(
    BASE_DIR,
    "prediction",
    "fuel_predictor_model.joblib"
)


# --------------------------------------------------
# Train model
# --------------------------------------------------

def train_and_save_model():

    print("[Model] Loading FuelCast data...")

    X, y, feature_cols = load_and_preprocess_data()

    print("\n[Model] Splitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print(f"[Model] Training samples: {len(X_train)}")
    print(f"[Model] Testing samples: {len(X_test)}")


    # --------------------------------------------------
    # XGBoost pipeline
    # --------------------------------------------------

    pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "model",
            XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        )
    ])


    print("\n[Model] Training XGBoost...")

    pipeline.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    print("\n[Model] Evaluating...")

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = mean_squared_error(
        y_test,
        y_pred
    ) ** 0.5

    r2 = r2_score(
        y_test,
        y_pred
    )


    print("\n========== MODEL RESULTS ==========")

    print(f"MAE  : {mae:.6f} kg/s")
    print(f"RMSE : {rmse:.6f} kg/s")
    print(f"R²   : {r2:.6f}")

    print("===================================\n")


    # --------------------------------------------------
    # Save model package
    # --------------------------------------------------

    model_package = {
        "model": pipeline,
        "features": feature_cols,
        "target": "ME_Total_MomentaryFuel",
        "unit": "kg/s",
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2)
    }


    joblib.dump(
        model_package,
        MODEL_SAVE_PATH
    )


    print(
        f"[Model] Saved successfully to:\n"
        f"{MODEL_SAVE_PATH}"
    )

    return model_package


# --------------------------------------------------
# Run training
# --------------------------------------------------

if __name__ == "__main__":
    train_and_save_model()