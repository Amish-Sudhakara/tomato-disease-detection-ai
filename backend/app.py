from fastapi import FastAPI, UploadFile, File
import shutil
import os

from backend.predict import predict_disease

app = FastAPI()

UPLOAD_FOLDER = "temp"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.get("/")
def home():
    return {"message": "Tomato Disease Prediction API Running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = f"{UPLOAD_FOLDER}/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_disease(file_path)

    os.remove(file_path)

    return result