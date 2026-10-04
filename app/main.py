from fastapi import FastAPI, HTTPException
from app.schemas import QuestionPair
from app.predictor import predictor

app = FastAPI(
    title="Quora Duplicate Question Detection",
    description="Predicts whether two questions are duplicates using the notebook's preprocessing and ANN pipeline.",
    version="1.0.0",
)

@app.get("/")
def root():
    return {"message": "Quora Duplicate Question Detection API is running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": predictor.ready}

@app.post("/predict")
def predict(data: QuestionPair):
    try:
        result = predictor.predict(data.question1, data.question2)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
