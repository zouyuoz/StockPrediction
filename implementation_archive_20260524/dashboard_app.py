from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import os

app = FastAPI()

# Setup templates
templates = Jinja2Templates(directory="templates")

CSV_FILE = "long_pred_test_result.csv"

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})

@app.get("/api/data")
async def get_data():
    if not os.path.exists(CSV_FILE):
        return {"error": "CSV file not found", "data": []}
    
    df = pd.read_csv(CSV_FILE)
    # Convert NaN to None for JSON compatibility
    data = df.where(pd.notnull(df), None).to_dict(orient="records")
    return {"data": data}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
