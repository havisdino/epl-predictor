from fastapi import FastAPI, HTTPException
import torch
from torch import nn

from models.mlp import MLPWithAttention

app = FastAPI()

# Example: adjust these dimensions as needed
INPUT_DIM = 10
HIDDEN_DIM = 32
OUTPUT_DIM = 1

model = MLPWithAttention(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
model.eval()

class PredictRequest(BaseModel):
    features: list[float]

class PredictResponse(BaseModel):
    prediction: float

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if len(request.features) != INPUT_DIM:
        raise HTTPException(status_code=400, detail=f"Expected {INPUT_DIM} features.")
    with torch.no_grad():
        x = torch.tensor(request.features, dtype=torch.float32)
        output = model(x)
        prediction = output.item()
    return PredictResponse(prediction=prediction)