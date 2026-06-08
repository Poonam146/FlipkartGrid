import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor

# 1. Load train and test data
print("Loading datasets...")
train_path = "dataset/train.csv"
test_path = "dataset/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# 2. Fill missing values
print("Imputing missing values...")
temp_median = train['Temperature'].median()
print(f"Imputing Temperature with train median: {temp_median}")

for df_temp in [train, test]:
    df_temp['RoadType'] = df_temp['RoadType'].fillna('Unknown')
    df_temp['Weather'] = df_temp['Weather'].fillna('Unknown')
    df_temp['Temperature'] = df_temp['Temperature'].fillna(temp_median)

# 3. Convert timestamp to datetime (incorporating the day offset)
print("Converting timestamp to datetime with day offset...")
for df_temp in [train, test]:
    # Convert H:M timestamp string to Timedelta
    td = pd.to_timedelta(df_temp['timestamp'] + ':00', errors='coerce')
    td = td.fillna(pd.Timedelta(seconds=0))
    # Combine with day starting from a base date (e.g. 2026-01-01)
    base_date = pd.to_datetime('2026-01-01')
    df_temp['datetime'] = base_date + pd.to_timedelta(df_temp['day'] - 1, unit='D') + td

# 4. Create features
print("Extracting features...")
for df_temp in [train, test]:
    df_temp['hour'] = df_temp['datetime'].dt.hour
    df_temp['dayofweek'] = df_temp['datetime'].dt.dayofweek
    df_temp['month'] = df_temp['datetime'].dt.month
    df_temp['is_weekend'] = df_temp['datetime'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Cyclic hour features
    df_temp['hour_sin'] = np.sin(2 * np.pi * df_temp['hour'] / 24.0)
    df_temp['hour_cos'] = np.cos(2 * np.pi * df_temp['hour'] / 24.0)
    
    # Rush hour feature (7-10 AM and 5-8 PM)
    df_temp['rush_hour'] = (((df_temp['hour'] >= 7) & (df_temp['hour'] <= 10)) | 
                            ((df_temp['hour'] >= 17) & (df_temp['hour'] <= 20))).astype(int)

# Frequency encoding for geohash
print("Creating geohash frequency encoding...")
geohash_counts = train['geohash'].value_counts()
train['geohash_freq'] = train['geohash'].map(geohash_counts)
test['geohash_freq'] = test['geohash'].map(geohash_counts).fillna(0)

# 5. Drop timestamp, helper datetime, and non-required features
# Features to KEEP:
# - Original features: geohash, RoadType, NumberofLanes, LargeVehicles, Landmarks, Temperature, Weather, day
# - Engineered features: geohash_freq, hour, dayofweek, month, is_weekend, hour_sin, hour_cos, rush_hour
keep_cols = [
    'geohash', 'RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks', 'Temperature', 'Weather', 'day',
    'geohash_freq', 'hour', 'dayofweek', 'month', 'is_weekend', 'hour_sin', 'hour_cos', 'rush_hour'
]

# Separate features and target
X = train[keep_cols].copy()
y = train['demand']
X_test = test[keep_cols].copy()

# Explicitly cast categorical columns to string format for CatBoost
cat_features = ['geohash', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
for col in cat_features:
    X[col] = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)

print(f"Features used for training ({len(X.columns)}): {list(X.columns)}")
print(f"Categorical features: {cat_features}")

# 1. Use 5-Fold Cross Validation
print("Running 5-Fold Cross Validation with CatBoost (Generalized Model)...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train))
test_preds_total = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n--- Training Fold {fold+1} ---")
    
    # Split training and validation folds
    X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx]
    X_test_fold = X_test.copy()
    
    # 7. Use CatBoost with specified hyperparameters
    model = CatBoostRegressor(
        iterations=5000,
        learning_rate=0.02,
        depth=6,
        l2_leaf_reg=10,
        random_strength=3,
        bagging_temperature=1,
        loss_function='RMSE',
        cat_features=cat_features,
        random_seed=42 + fold,
        verbose=300
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=300
    )
    
    # Predict validation fold
    val_preds = model.predict(X_val)
    oof_preds[val_idx] = val_preds
    
    # Predict test fold
    test_preds_fold = model.predict(X_test_fold)
    test_preds_total += test_preds_fold
    
    fold_r2 = r2_score(y_val, val_preds)
    print(f"Fold {fold+1} Validation R2 Score: {fold_r2:.6f}")

# Calculate overall CV R2 score
cv_r2 = r2_score(y, oof_preds)
print(f"\n======================================")
print(f"Overall Generalized CatBoost OOF CV R2 Score: {cv_r2:.6f}")
print(f"======================================\n")

# Average the test predictions across all folds
test_preds_avg = test_preds_total / 5.0

# 11. Create submission_catboost_generalized.csv with columns: Index, demand
submission = pd.DataFrame({
    'Index': test['Index'],
    'demand': test_preds_avg
})

# 12. Save submission_catboost_generalized.csv in project root
submission_path = "submission_catboost_generalized.csv"
submission.to_csv(submission_path, index=False)
print(f"Saved submission to {submission_path}")

# Verify file exists and has correct rows
if os.path.exists(submission_path):
    sub_df = pd.read_csv(submission_path)
    print(f"Submission verification: shape={sub_df.shape}, cols={list(sub_df.columns)}")
    print(sub_df.head())
else:
    print("Error: Submission file was not created!")
