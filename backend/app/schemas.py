from pydantic import BaseModel
from typing import Dict, Any

# Generic request: accept key-value pairs for features
class PredictRequest(BaseModel):
    features: Dict[str, Any]

class PredictResponse(BaseModel):
    predicted_price: float
    unit: str = "INR"
    note: str = "Price is a point estimate from the trained model"
