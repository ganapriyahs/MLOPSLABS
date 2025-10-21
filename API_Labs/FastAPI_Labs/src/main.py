from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data 

app = FastAPI()

# California housing has 8 features
class HousingData(BaseModel):
    MedInc: float       # median income in block group
    HouseAge: float     # median house age in block group
    AveRooms: float     # average number of rooms
    AveBedrms: float    # average number of bedrooms
    Population: float   # block group population
    AveOccup: float     # average house occupancy
    Latitude: float     # block latitude
    Longitude: float    # block longitude

class HousingResponse(BaseModel):
    predicted_value: float

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=HousingResponse)
async def predict_housing(housing_features: HousingData):
    try:
        features = [[
            housing_features.MedInc,
            housing_features.HouseAge,
            housing_features.AveRooms,
            housing_features.AveBedrms,
            housing_features.Population,
            housing_features.AveOccup,
            housing_features.Latitude,
            housing_features.Longitude
        ]]

        prediction = predict_data(features) 
        return HousingResponse(predicted_value=float(prediction[0]))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
