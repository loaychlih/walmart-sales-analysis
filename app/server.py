from fastapi import FastAPI
import pandas as pd
import joblib
from pydantic import BaseModel

# Load the trained model
model = joblib.load('/home/loay/walmart_sales_analysis/app/walmart_sales_model.pkl')

app = FastAPI()

# Define the input data model
class SalesData(BaseModel):
    Store: int                
    Holiday_Flag: int         
    Temperature: float         
    Fuel_Price: float          
    CPI: float                 
    Unemployment: float        
    Month: int                 
    Year: int                  
    Week: int                  

@app.get("/")
def read_root():
    return {"message": "Welcome to the Walmart Sales Prediction API"}

@app.post("/predict/")
def predict(data: SalesData):
    # Convert input data to a DataFrame
    input_data = pd.DataFrame([data.dict()])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    return {"prediction": prediction.tolist()}
