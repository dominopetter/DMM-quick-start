import pickle
import uuid
import datetime
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.tree import DecisionTreeRegressor
from domino_data_capture.data_capture_client import DataCaptureClient

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
    mlflow.log_param("model_type", "DecisionTreeRegressor")
    mlflow.sklearn.log_model(model, "price_prediction_model")
    run_id = mlflow.active_run().info.run_id

# Register the model in MLflow registry
model_uri = f"runs:/{run_id}/price_prediction_model"
mlflow.register_model(model_uri, "HousePricePredictionModel")

def predict_price(bedrooms, bathrooms,sqft_living,sqft_lot,floors,
                    waterfront,view,condition,grade,sqft_above,sqft_basement,
                    yr_built,yr_renovated,zipcode,lat,long, sqft_living15,
                    sqft_lot15):
    
    feature_values = [bedrooms, bathrooms,sqft_living,sqft_lot,floors,
                    waterfront,view,condition,grade,sqft_above,sqft_basement,
                    yr_built,yr_renovated,zipcode,lat,long,
                    sqft_living15,sqft_lot15]
    
    price_prediction = model.predict([feature_values])
    
    # Record eventID and current time
    event_id = uuid.uuid4()
    event_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    # Capture this prediction event so Domino can track and monitor them
    data_capture_client.capturePrediction(
        feature_values,
        price_prediction,
        event_id=event_id,
        timestamp=event_time,
    )
    
    return dict(prediction=price_prediction[0])

# Transition the registered model to "Staging"
client = MlflowClient()
client.transition_model_version_stage(
    name="HousePricePredictionModel",
    version=1,
    stage="Staging"
)

print(f"Model registered and moved to Staging with URI: {model_uri}")