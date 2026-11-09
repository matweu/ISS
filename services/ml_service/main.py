from fastapi import FastAPI, HTTPException, Query
 
from api_handler import FastAPIHandler
 
app = FastAPI(
    title="Mobile Price Prediction Service",
    description="Сервис предсказания ценовой категории телефона",
    version="1.0.0"
)
 
handler = FastAPIHandler()
 
@app.get("/")
def root():
    return {"Hello": "World"}
 
@app.post("/api/prediction")
def prediction(
    features: dict[str, int | float],
    item_id: int = Query(...)
):
 
    try:
 
        result = handler.predict(features)
 
        return {
            "item_id": item_id,
            "predict": result
        }
 
    except ValueError as error:
 
        raise HTTPException(
            status_code=422,
            detail=str(error)
        ) from error
