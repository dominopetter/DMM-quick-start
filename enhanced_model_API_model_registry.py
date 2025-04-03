import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd

# Feature list for context (not actually used in training here)
features = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
            "waterfront", "view", "condition", "grade", "sqft_above",
            "sqft_basement", "yr_built", "yr_renovated", "zipcode", "lat", "long",
            "sqft_living15", "sqft_lot15"]

# Generate multiple runs with varied hyperparameters
for depth in [3, 5, 7]:
    for min_samples_split in [2, 5]:
        with mlflow.start_run():
            # Create model with different hyperparameters
            model = DecisionTreeRegressor(max_depth=depth, min_samples_split=min_samples_split)

            # Dummy training for demo (no real data needed)
            X_dummy = [[0] * len(features)] * 10
            y_dummy = list(range(10))
            model.fit(X_dummy, y_dummy)

            # Log parameters to generate branching lines
            mlflow.log_param("model_type", "DecisionTreeRegressor")
            mlflow.log_param("max_depth", depth)
            mlflow.log_param("min_samples_split", min_samples_split)

            # Log varied metrics for visual diversity
            rmse = 100000 / depth + np.random.randint(-1000, 1000)
            r2 = 0.5 + (depth * 0.05) - (min_samples_split * 0.01)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2_score", r2)

            # Log the model
            mlflow.sklearn.log_model(model, "price_prediction_model")

            # Log dummy feature importances as an artifact
            importances = np.random.dirichlet(np.ones(len(features)), size=1)[0]
            importance_df = pd.DataFrame({"feature": features, "importance": importances})
            importance_file = f"feature_importances_depth{depth}_split{min_samples_split}.csv"
            importance_df.to_csv(importance_file, index=False)
            mlflow.log_artifact(importance_file)
