import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import lightgbm as lgb

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

# 4. Create timestamp features
print("Extracting datetime features...")
for df_temp in [train, test]:
    df_temp['hour'] = df_temp['datetime'].dt.hour
    df_temp['dayofweek'] = df_temp['datetime'].dt.dayofweek
    df_temp['month'] = df_temp['datetime'].dt.month
    df_temp['week_of_year'] = df_temp['datetime'].dt.isocalendar().week.astype(int)
    df_temp['quarter'] = df_temp['datetime'].dt.quarter
    df_temp['day_of_month'] = df_temp['datetime'].dt.day
    df_temp['is_weekend'] = df_temp['datetime'].dt.dayofweek.isin([5, 6]).astype(int)

# 6. Create cyclic encoding
print("Creating cyclic encoding features...")
for df_temp in [train, test]:
    df_temp['hour_sin'] = np.sin(2 * np.pi * df_temp['hour'] / 24.0)
    df_temp['hour_cos'] = np.cos(2 * np.pi * df_temp['hour'] / 24.0)

# 7. Create rush_hour feature (7-10 AM and 5-8 PM)
print("Creating rush_hour feature...")
for df_temp in [train, test]:
    df_temp['rush_hour'] = (((df_temp['hour'] >= 7) & (df_temp['hour'] <= 10)) | 
                            ((df_temp['hour'] >= 17) & (df_temp['hour'] <= 20))).astype(int)

# 8. Create interaction features (hour * RoadType, hour * Weather)
print("Creating interaction features...")
for df_temp in [train, test]:
    df_temp['hour_RoadType'] = df_temp['hour'].astype(str) + '_' + df_temp['RoadType'].astype(str)
    df_temp['hour_Weather'] = df_temp['hour'].astype(str) + '_' + df_temp['Weather'].astype(str)

# 9. Create temperature bins
print("Binning Temperature...")
# Bin training Temperature and retrieve edges
train['temp_bin'], bin_edges = pd.cut(train['Temperature'], bins=5, retbins=True, labels=False)
train['temp_bin'] = train['temp_bin'].astype(str)
# Apply the same edges to the test Temperature
test['temp_bin'] = pd.cut(test['Temperature'], bins=bin_edges, labels=False, include_lowest=True)
test['temp_bin'] = test['temp_bin'].fillna(train['temp_bin'].astype(float).median()).astype(int).astype(str)

# 3. Add geohash frequency encoding
print("Creating geohash frequency encoding...")
geohash_counts = train['geohash'].value_counts()
train['geohash_freq'] = train['geohash'].map(geohash_counts)
test['geohash_freq'] = test['geohash'].map(geohash_counts).fillna(0)

# Drop timestamp and helper datetime
print("Dropping timestamp and helper datetime...")
train = train.drop(columns=['timestamp', 'datetime'])
test = test.drop(columns=['timestamp', 'datetime'])

# Separate target and features
X = train.drop(columns=['Index', 'demand'])
y = train['demand']
X_test = test.drop(columns=['Index'])

# Explicitly cast categorical columns to pandas category dtype
cat_features = ['geohash', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather', 'hour_RoadType', 'hour_Weather', 'temp_bin']
for col in cat_features:
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# 1. Use 5-Fold Cross Validation
print("Running 5-Fold Cross Validation with LightGBM...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train))
test_preds_total = np.zeros(len(test))
feature_importances = np.zeros(X.shape[1] + 1) # +1 for geohash_te feature created later

feature_names = None

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n--- Training Fold {fold+1} ---")
    
    # Split training and validation folds
    X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx]
    X_test_fold = X_test.copy()
    
    # 4. Add geohash target mean encoding using out-of-fold strategy
    geohash_means = y_train.groupby(X_train['geohash']).mean()
    global_mean = y_train.mean()
    
    X_train['geohash_te'] = X_train['geohash'].map(geohash_means).fillna(global_mean)
    X_val['geohash_te'] = X_val['geohash'].map(geohash_means).fillna(global_mean)
    X_test_fold['geohash_te'] = X_test_fold['geohash'].map(geohash_means).fillna(global_mean)
    
    # Get feature names if not saved yet
    if feature_names is None:
        feature_names = list(X_train.columns)
    
    # Instantiate LightGBM Regressor
    model = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=10,
        num_leaves=127,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbosity=-1
    )
    
    # Train with early stopping callback
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        categorical_feature=cat_features,
        callbacks=[
            lgb.early_stopping(stopping_rounds=200, verbose=False),
            lgb.log_evaluation(300)
        ]
    )
    
    # Predict validation fold
    val_preds = model.predict(X_val)
    oof_preds[val_idx] = val_preds
    
    # Predict test fold
    test_preds_fold = model.predict(X_test_fold)
    test_preds_total += test_preds_fold
    
    # Feature importances
    feature_importances += model.feature_importances_
    
    fold_r2 = r2_score(y_val, val_preds)
    print(f"Fold {fold+1} Validation R2 Score: {fold_r2:.6f}")

# Calculate overall CV R2 score
cv_r2 = r2_score(y, oof_preds)
print(f"\n======================================")
print(f"Overall LightGBM OOF CV R2 Score: {cv_r2:.6f}")
print(f"======================================\n")

# Average feature importances
feature_importances /= 5.0
fi_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("Top 15 Feature Importances:")
print(fi_df.head(15).to_string(index=False))

# Average test predictions
test_preds_avg = test_preds_total / 5.0

# Save OOF predictions
np.save("lgbm_oof.npy", oof_preds)


# Generate submission_lgbm.csv
submission = pd.DataFrame({
    'Index': test['Index'],
    'demand': test_preds_avg
})

submission_path = "submission_lgbm.csv"
submission.to_csv(submission_path, index=False)
print(f"Saved submission to {submission_path}")

# Verify file exists and has correct rows
if os.path.exists(submission_path):
    sub_df = pd.read_csv(submission_path)
    print(f"Submission verification: shape={sub_df.shape}, cols={list(sub_df.columns)}")
    print(sub_df.head())
else:
    print("Error: Submission file was not created!")
