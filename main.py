import os
import ee
import json
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from google.cloud import storage
from fastai.learner import load_learner
from fastai.vision.core import PILImage
from io import BytesIO

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# --- GEE Authentication and Initialization ---
GEE_SERVICE_ACCOUNT_KEY_PATH = os.getenv("GEE_SERVICE_ACCOUNT_KEY_PATH")
if not GEE_SERVICE_ACCOUNT_KEY_PATH or not os.path.exists(GEE_SERVICE_ACCOUNT_KEY_PATH):
    raise ValueError("GEE_SERVICE_ACCOUNT_KEY_PATH not set or file not found in .env")

try:
    ee.Authenticate(json_key_file=GEE_SERVICE_ACCOUNT_KEY_PATH)
    ee.Initialize()
    print("Google Earth Engine initialized successfully.")
except Exception as e:
    print(f"Error initializing Google Earth Engine: {e}")
    # Consider more robust error handling for production, e.g., logging or exiting

# --- Google Cloud Storage Client (for exports) ---
GCS_EXPORT_BUCKET_NAME = os.getenv("GCS_EXPORT_BUCKET_NAME")
if GCS_EXPORT_BUCKET_NAME:
    try:
        storage_client = storage.Client()
        print(f"Google Cloud Storage client initialized for bucket: {GCS_EXPORT_BUCKET_NAME}")
    except Exception as e:
        print(f"Error initializing Google Cloud Storage client: {e}")
        storage_client = None
else:
    print("GCS_EXPORT_BUCKET_NAME not set in .env. Export functionality will be limited.")
    storage_client = None

# --- CORS Configuration ---
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # Add your production React app URL(s) here if deployed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load FastAI Model ---
# Ensure the path to your FastAI model is correct
# For example, if 'floor_detector2.pkl' is in a directory named 'models' at the same level as main.py:
# learn = load_learner("models/floor_detector2.pkl")
# If 'floor_detector2.pkl' is in a subfolder named 'app' within your backend:
try:
    learn = load_learner("app/floor_detector2.pkl")
    print("FastAI model 'floor_detector2.pkl' loaded successfully.")
except Exception as e:
    print(f"Error loading FastAI model: {e}")
    learn = None # Set to None if model loading fails
    # Consider raising an exception or handling this more gracefully if the model is critical

# --- FastAPI Endpoints ---

@app.get("/")
async def root():
    return {"message": "FastAPI Earth Engine and Floor Detector Integration is running!"}

@app.post("/get_gee_map_layer")
async def get_gee_map_layer(data: dict):
    layer_type = data.get("layer_type", "default")
    aoi_geojson = data.get("aoi_geojson")
    start_date = data.get("start_date")
    end_date = data.get("end_date")

    try:
        collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR') \
            .filterDate(start_date, end_date) \
            .filterBounds(ee.Geometry.Polygon(aoi_geojson['coordinates']))

        base_image = collection.median()

        nir = base_image.select('B5')
        red = base_image.select('B4')
        ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')

        highlight_mask = ndvi.lt(0.2).And(ndvi.gt(-1)).selfMask()

        highlight_vis_params = {'min': 0, 'max': 1, 'palette': ['00000000', 'FF0000FF']}

        image_to_display = base_image.visualize(bands=['B4', 'B3', 'B2'], min=0, max=3000, gamma=1.4)
        highlight_layer = highlight_mask.visualize(**highlight_vis_params)

        map_id_highlight = highlight_layer.getMapId()
        map_id_base = image_to_display.getMapId()

        return {
            "highlight_mapid": map_id_highlight['mapid'],
            "highlight_token": map_id_highlight['token'],
            "base_mapid": map_id_base['mapid'],
            "base_token": map_id_base['token'],
            "tile_url_format": "https://earthengine.googleapis.com/v1alpha/projects/earthengine-public/maps/{mapid}/tiles/{z}/{x}/{y}?token={token}"
        }

    except ee.EEException as e:
        raise HTTPException(status_code=500, detail=f"Earth Engine error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/export_gee_image")
async def export_gee_image(export_request: dict):
    if not storage_client:
        raise HTTPException(status_code=503, detail="Google Cloud Storage client not initialized. Cannot export.")

    aoi_geojson = export_request.get("aoi_geojson")
    start_date = export_request.get("start_date")
    end_date = export_request.get("end_date")
    file_name_prefix = export_request.get("file_name_prefix", "gee_export")
    scale = export_request.get("scale", 30)
    file_format = export_request.get("file_format", "GeoTIFF")

    if not aoi_geojson:
        raise HTTPException(status_code=400, detail="AOI GeoJSON is required for export.")
    if not (start_date and end_date):
        raise HTTPException(status_code=400, detail="Start and end dates are required.")

    try:
        collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR') \
            .filterDate(start_date, end_date) \
            .filterBounds(ee.Geometry.Polygon(aoi_geojson['coordinates']))
        base_image = collection.median()

        nir = base_image.select('B5')
        red = base_image.select('B4')
        ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
        highlight_mask = ndvi.lt(0.2).And(ndvi.gt(-1)).selfMask()

        image_to_export = ndvi.updateMask(highlight_mask)

        geometry_ee = ee.Geometry.Polygon(aoi_geojson['coordinates'])

        task = ee.batch.Export.image.toCloudStorage(
            image=image_to_export,
            description=file_name_prefix,
            bucket=GCS_EXPORT_BUCKET_NAME,
            fileNamePrefix=file_name_prefix,
            scale=scale,
            region=geometry_ee.toGeoJSONString(),
            fileFormat=file_format,
        )
        task.start()

        return {"message": "Export task started.", "task_id": task.id}

    except ee.EEException as e:
        raise HTTPException(status_code=500, detail=f"Earth Engine export error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error during export: {str(e)}")


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if learn is None:
        raise HTTPException(status_code=500, detail="FastAI model not loaded.")
    
    image_bytes = await file.read()
    img = PILImage.create(BytesIO(image_bytes))
    img = img.resize((224, 224))
    pred_class, pred_idx, probs = learn.predict(img)

    return {
        "predicted_class": str(pred_class),
        "class_index": int(pred_idx),
        "probabilities": [float(p) for p in probs]
    }