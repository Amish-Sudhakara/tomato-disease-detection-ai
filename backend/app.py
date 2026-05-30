import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from backend.config import UPLOAD_FOLDER
from backend.model_loader import load_all_models
from backend.predict import predict_disease

@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    # System pre-loading routines
    load_all_models()
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    yield

app = FastAPI(title="Unified Tomato Disease Classifier API", lifespan=lifespan)

@app.get("/")
def home():
    return {"message": "Unified Tomato Disease Prediction API Running"}

@app.post("/predict/{model_type}")
async def predict(model_type: str, file: UploadFile = File(...)):
    if model_type not in ["mobilenet", "efficientnet"]:
        raise HTTPException(status_code=400, detail="Invalid model selection. Use 'mobilenet' or 'efficientnet'.")
        
    if not file.filename:
        raise HTTPException(status_code=400, detail="Invalid file descriptor.")

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = predict_disease(file_path, model_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Pipeline Exception: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    return result