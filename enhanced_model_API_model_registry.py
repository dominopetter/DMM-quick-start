import pickle
import uuid
import datetime
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.tree import DecisionTreeRegressor
from domino_data_capture.data_capture_client import DataCaptureClient
import pandas as pd  # Added to handle DataFrame creation for feature importances

features = ["bedrooms", "bathrooms","sqft_living","sqft_lot","floors",
            "waterfront","view","condition","grade", "sqft_above",
            "sqft_basement","yr_built","yr_renovated","zipcode","lat","long",
            "sqft_living15","sqft_lot15"]

target = ["price"]

data_capture_client = DataCaptureClient(features, target)

model_file_name = "price_dt_py.sav"
model = pickle.load(open(model_file_name, 'rb'))

# Start MLflow logging and register model
with mlflow.start_run():
    # Log base parameter
    mlflow.log_param("model_type", "DecisionTreeRegressor")

    # Log model hyperparameters
    # (Added for demo purposes to show more tracked parameters)
    mlflow.log_param("max_depth", model.get_params().get("max_depth"))
    mlflow.log_param("min_samples_split", model.get_params().get("min_samples_split"))

    # Log the model artifact
    mlflow.sklearn.log_model(model, "price_prediction_model")
    run_id = mlflow.active_run().info.run_id

    # Log demo metrics over steps to create a nice curve (Added for visualization)
    for step in range(10):
        mlflow.log_metric("rmse", 300000 / (step + 1), step=step)
        mlflow.log_metric("r2_score", 0.5 + 0.05 * step, step=step)

    # Log feature importances as a CSV artifact (Added to show artifacts)
    importances = model.feature_importances_
    importance_df = pd.DataFrame({"feature": features, "importance": importances})
    importance_df.to_csv("feature_importances.csv", index=False)
    mlflow.log_artifact("feature_importances.csv")

# Register the model in MLflow registry
model_uri = f"runs:/{run_id}/price_prediction_model"
mlflow.register_model(model_uri, "price_prediction_model_registry")
