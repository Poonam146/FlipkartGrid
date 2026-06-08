import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

def main():
    # 1. Load ground truth demand from train.csv
    print("Loading train dataset...")
    train_path = "dataset/train.csv"
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found!")
        return
    train = pd.read_csv(train_path)
    y_true = train['demand'].values
    
    # 2. Load OOF predictions
    print("Loading out-of-fold predictions...")
    cb_oof_path = "catboost_oof.npy"
    lgb_oof_path = "lgbm_oof.npy"
    
    if not os.path.exists(cb_oof_path) or not os.path.exists(lgb_oof_path):
        print("Error: OOF prediction files (npy) not found! Make sure both solution.py and solution_lgbm.py have run to completion.")
        return
        
    cb_oof = np.load(cb_oof_path)
    lgb_oof = np.load(lgb_oof_path)
    
    # 3. Try weights
    candidate_weights = [
        (0.7, 0.3),
        (0.6, 0.4),
        (0.5, 0.5)
    ]
    
    best_r2 = -float('inf')
    best_weights = None
    
    print("\nEvaluating validation ensemble R2 scores:")
    for w_cb, w_lgb in candidate_weights:
        ensemble_oof = w_cb * cb_oof + w_lgb * lgb_oof
        r2 = r2_score(y_true, ensemble_oof)
        print(f"Weights: {w_cb:.1f} CatBoost + {w_lgb:.1f} LightGBM -> Validation R2 Score: {r2:.6f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_weights = (w_cb, w_lgb)
            
    print(f"\nBest Ensemble Weights: {best_weights[0]:.1f} CatBoost + {best_weights[1]:.1f} LightGBM")
    print(f"Best Validation R2 Score: {best_r2:.6f}")
    
    # 4. Load test predictions
    cb_sub_path = "submission_v2.csv"
    lgb_sub_path = "submission_lgbm.csv"
    
    if not os.path.exists(cb_sub_path) or not os.path.exists(lgb_sub_path):
        print(f"Error: {cb_sub_path} or {lgb_sub_path} not found! Make sure training runs generated them.")
        return
        
    print(f"\nLoading test predictions from {cb_sub_path} and {lgb_sub_path}...")
    cb_sub = pd.read_csv(cb_sub_path)
    lgb_sub = pd.read_csv(lgb_sub_path)
    
    # 5. Generate ensemble_submission.csv
    w_cb, w_lgb = best_weights
    ensemble_preds = w_cb * cb_sub['demand'] + w_lgb * lgb_sub['demand']
    
    ensemble_sub = pd.DataFrame({
        'Index': cb_sub['Index'],
        'demand': ensemble_preds
    })
    
    out_path = "ensemble_submission.csv"
    ensemble_sub.to_csv(out_path, index=False)
    print(f"Saved ensemble submission to {out_path}")
    
    # Verify file
    if os.path.exists(out_path):
        sub_df = pd.read_csv(out_path)
        print(f"Ensemble submission verification: shape={sub_df.shape}, cols={list(sub_df.columns)}")
        print(sub_df.head())
        
if __name__ == "__main__":
    main()
