Flipkart GRiD 6.0: Flipkart Gridlock Hackathon 2.0 - Traffic Demand Prediction
=============================================================================

Problem:
  Traffic Demand Prediction (predict passenger demand at spatiotemporal buckets)

Models Used:
  - CatBoost Regressor
  - LightGBM Regressor

Final Approach:
  An ensemble model blending predictions from CatBoost (40%) and LightGBM (60%).
  The ensemble weights (0.4 CatBoost + 0.6 LightGBM) were selected based on
  out-of-fold validation performance, which showed the strongest generalization.

Feature Engineering:
  1. Missing Value Handling:
     - RoadType filled with "Unknown"
     - Weather filled with "Unknown"
     - Temperature filled with training median to avoid target leak
  2. Timestamp Features:
     - hour, dayofweek, month, week_of_year, quarter, day_of_month, is_weekend
  3. Cyclic Hour Encoding:
     - hour_sin = sin(2 * pi * hour / 24)
     - hour_cos = cos(2 * pi * hour / 24)
  4. Rush Hour Feature:
     - Binary indicator for peak hours (7-10 AM and 5-8 PM)
  5. Geohash Frequency Encoding:
     - geohash mapped to its historical occurrence count in training set

Evaluation Metric:
  R2 Score
