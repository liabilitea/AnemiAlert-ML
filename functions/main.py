import os
import io
import logging
import traceback
import time
from datetime import datetime
from typing import Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import requests
import pytz

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, firestore, db
from firebase_functions import https_fn
import functions_framework

from inference import get_predictor
from preprocessing import validate_image, validate_ppg_values

# Initialize Firebase Admin SDK with database URL and storage bucket
if not firebase_admin._apps:
    firebase_admin.initialize_app(options={
        'databaseURL': 'https://anemialert-7b776-default-rtdb.asia-southeast1.firebasedatabase.app',
        'storageBucket': 'anemialert-7b776.appspot.com'
    })

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Hemoglobin Prediction API",
    description="Non-invasive hemoglobin prediction using eye image and PPG signals",
    version="1.0.0"
)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: bool


class PredictionResponse(BaseModel):
    """Prediction response."""
    success: bool
    hemoglobin_prediction: Optional[float] = None
    eye_hemoglobin: Optional[float] = None
    ppg_hemoglobin: Optional[float] = None
    weights: Optional[dict] = None
    segmentation_stats: Optional[dict] = None
    error: Optional[str] = None


def determine_anemia_status(hemoglobin_gdl: float, age: int, sex: str, is_pregnant: bool = False, trimester: int = 2) -> str:
    """
    Determine anemia status based on WHO guidelines.
    
    Args:
        hemoglobin_gdl: Hemoglobin in g/dL
        age: Age in years
        sex: "Male" or "Female"
        is_pregnant: True if pregnant
        trimester: 1, 2, or 3 (only used if is_pregnant=True)
    
    Returns:
        "No anemia", "Mild anemia", "Moderate anemia", or "Severe anemia"
    """
    # Convert g/dL to g/L for WHO standards
    hb_gl = hemoglobin_gdl * 10
    
    # Pregnancy takes precedence
    if is_pregnant:
        if trimester == 1 or trimester == 3:
            if hb_gl >= 110:
                return "No anemia"
            elif hb_gl >= 100:
                return "Mild anemia"
            elif hb_gl >= 70:
                return "Moderate anemia"
            else:
                return "Severe anemia"
        else:  # trimester == 2
            if hb_gl >= 105:
                return "No anemia"
            elif hb_gl >= 95:
                return "Mild anemia"
            elif hb_gl >= 70:
                return "Moderate anemia"
            else:
                return "Severe anemia"
    
    # Age-based thresholds
    if age < 2:  # 6-23 months
        if hb_gl >= 105:
            return "No anemia"
        elif hb_gl >= 95:
            return "Mild anemia"
        elif hb_gl >= 70:
            return "Moderate anemia"
        else:
            return "Severe anemia"
    
    elif age < 5:  # 24-59 months
        if hb_gl >= 110:
            return "No anemia"
        elif hb_gl >= 100:
            return "Mild anemia"
        elif hb_gl >= 70:
            return "Moderate anemia"
        else:
            return "Severe anemia"
    
    elif age < 12:  # 5-11 years
        if hb_gl >= 115:
            return "No anemia"
        elif hb_gl >= 110:
            return "Mild anemia"
        elif hb_gl >= 80:
            return "Moderate anemia"
        else:
            return "Severe anemia"
    
    elif age < 15:  # 12-14 years
        if hb_gl >= 120:
            return "No anemia"
        elif hb_gl >= 110:
            return "Mild anemia"
        elif hb_gl >= 80:
            return "Moderate anemia"
        else:
            return "Severe anemia"
    
    else:  # 15+ years (adults)
        if sex == "Male":
            if hb_gl >= 130:
                return "No anemia"
            elif hb_gl >= 110:
                return "Mild anemia"
            elif hb_gl >= 80:
                return "Moderate anemia"
            else:
                return "Severe anemia"
        else:  # Female, non-pregnant
            if hb_gl >= 120:
                return "No anemia"
            elif hb_gl >= 110:
                return "Mild anemia"
            elif hb_gl >= 80:
                return "Moderate anemia"
            else:
                return "Severe anemia"


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    try:
        logger.info("Starting up application...")
        predictor = get_predictor()
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Hemoglobin Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Verifies that models are loaded and ready.
    """
    try:
        predictor = get_predictor()
        models_loaded = (
            predictor.unet_model is not None and
            predictor.eye_model is not None and
            predictor.ppg_model is not None
        )
        
        return HealthResponse(
            status="healthy" if models_loaded else "unhealthy",
            models_loaded=models_loaded
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            models_loaded=False
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict_hemoglobin(
    eye_image: UploadFile = File(..., description="Eye image (JPEG/PNG)"),
    ir_value: float = Form(..., description="Infrared sensor value"),
    red_value: float = Form(..., description="Red light sensor value"),
    age: int = Form(..., description="Patient age", ge=0, le=120),
    gender: str = Form(..., description="Patient gender (Male/Female)")
):
    """
    Predict hemoglobin level from eye image and PPG values.
    This is the standalone API endpoint (for testing).
    """
    try:
        logger.info(f"Received prediction request: age={age}, gender={gender}")
        
        # Validate gender
        if gender not in ["Male", "Female", "male", "female"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid gender: {gender}. Must be 'Male' or 'Female'"
            )
        
        # Validate PPG values
        ppg_valid, ppg_error = validate_ppg_values(ir_value, red_value)
        if not ppg_valid:
            raise HTTPException(status_code=400, detail=ppg_error)
        
        # Read and validate image
        try:
            image_bytes = await eye_image.read()
            image_pil = Image.open(io.BytesIO(image_bytes))
            
            if image_pil.mode != 'RGB':
                image_pil = image_pil.convert('RGB')
            
            image_np = np.array(image_pil)
            
        except Exception as e:
            logger.error(f"Failed to read image: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # Validate image
        img_valid, img_error = validate_image(image_np)
        if not img_valid:
            raise HTTPException(status_code=400, detail=img_error)
        
        logger.info(f"Image loaded: shape={image_np.shape}, dtype={image_np.dtype}")
        
        # Get predictor and run inference
        predictor = get_predictor()
        
        result = predictor.predict(
            image=image_np,
            ir_value=ir_value,
            red_value=red_value,
            age=age,
            gender=gender
        )
        
        logger.info(f"Prediction successful: Hb={result['hemoglobin_prediction']:.2f} g/dL")
        
        return PredictionResponse(
            success=True,
            **result
        )
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        logger.error(traceback.format_exc())
        
        return PredictionResponse(
            success=False,
            error=f"Internal server error: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc)
        }
    )

# FIREBASE CLOUD FUNCTIONS - INTEGRATED VERSION

@functions_framework.http
def predict(request):
    """
    Firebase Cloud Functions entry point for standalone API.
    Routes HTTP requests to the FastAPI app.
    """
    import asyncio
    
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)
    
    cors_headers = {'Access-Control-Allow-Origin': '*'}
    
    try:
        path = request.path
        if path.startswith('/predict'):
            path = path[8:]
        if not path:
            path = '/'
        
        scope = {
            'type': 'http',
            'asgi': {'version': '3.0'},
            'http_version': '1.1',
            'method': request.method,
            'scheme': request.scheme,
            'path': path,
            'query_string': request.query_string,
            'root_path': '',
            'headers': [
                (k.lower().encode(), v.encode()) 
                for k, v in request.headers.items()
            ],
            'server': (request.host.split(':')[0], 443),
        }
        
        async def receive():
            return {
                'type': 'http.request',
                'body': request.get_data(),
                'more_body': False,
            }
        
        response_data = {
            'status': 200,
            'headers': [],
            'body': []
        }
        
        async def send(message):
            if message['type'] == 'http.response.start':
                response_data['status'] = message['status']
                response_data['headers'] = message.get('headers', [])
            elif message['type'] == 'http.response.body':
                body = message.get('body', b'')
                if body:
                    response_data['body'].append(body)
        
        asyncio.run(app(scope, receive, send))
        
        status_code = response_data['status']
        response_body = b''.join(response_data['body'])
        
        response_headers = dict(cors_headers)
        for name, value in response_data['headers']:
            header_name = name.decode() if isinstance(name, bytes) else name
            header_value = value.decode() if isinstance(value, bytes) else value
            response_headers[header_name] = header_value
        
        return (response_body, status_code, response_headers)
    
    except Exception as e:
        logger.error(f"Cloud Function error: {e}")
        logger.error(traceback.format_exc())
        
        import json
        error_response = {'success': False, 'error': str(e)}
        return (
            json.dumps(error_response).encode(),
            500,
            {**cors_headers, 'Content-Type': 'application/json'}
        )


@https_fn.on_call()
def predict_hemoglobin_integrated(req: https_fn.CallableRequest):
    """
    Integrated Firebase Callable Function.
    Reads from Firebase, runs prediction, saves results automatically.
    
    Input:
        record_id: The key from /data/{record_id} in Realtime Database
    
    Returns:
        Prediction results + result_id where it was saved
    """
    try:
        logger.info("=== Starting integrated prediction ===")
        
        # Get input parameter
        record_id = req.data.get('record_id')
        if not record_id:
            raise https_fn.HttpsError(
                code=https_fn.FunctionsErrorCode.INVALID_ARGUMENT,
                message='record_id is required'
            )
        
        logger.info(f"Processing record: {record_id}")
        
        # READ DATA FROM REALTIME DATABASE
        data_ref = db.reference(f'data/{record_id}')
        data = data_ref.get()
        
        if not data:
            raise https_fn.HttpsError(
                code=https_fn.FunctionsErrorCode.NOT_FOUND,
                message=f'Data record {record_id} not found'
            )
        
        patient_id = data.get('Patient_id')
        image_url = data.get('Conjunctiva_image')
        ir_value = data.get('Sensor_ir')
        red_value = data.get('Sensor_red')
        heart_rate = data.get('Sensor_bpm', 0)
        blood_oxygen = data.get('Sensor_spo2', 0)
        
        logger.info(f"Found patient: {patient_id}")
        logger.info(f"PPG values - IR: {ir_value}, Red: {red_value}")
        
        # READ PATIENT INFO FROM FIRESTORE
        firestore_client = firestore.client()
        patient_doc = firestore_client.collection('patient').document(patient_id).get()
        
        if not patient_doc.exists:
            raise https_fn.HttpsError(
                code=https_fn.FunctionsErrorCode.NOT_FOUND,
                message=f'Patient {patient_id} not found in Firestore'
            )
        
        patient_data = patient_doc.to_dict()
        age = patient_data.get('Patient_age')
        gender = patient_data.get('Patient_sex')
        pregnancy_status = patient_data.get('Pregnancy_status', 'No')
        
        logger.info(f"Patient info - Age: {age}, Gender: {gender}")
        
        # DOWNLOAD IMAGE FROM STORAGE URL
        logger.info(f"Downloading image from: {image_url}")
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        image_pil = Image.open(io.BytesIO(response.content))
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        image_np = np.array(image_pil)
        
        logger.info(f"Image loaded: shape={image_np.shape}")
        
        # RUN ML PREDICTION
        logger.info("Running ML prediction...")
        predictor = get_predictor()
        
        result = predictor.predict(
            image=image_np,
            ir_value=float(ir_value),
            red_value=float(red_value),
            age=int(age),
            gender=gender
        )
        
        hemoglobin = result['hemoglobin_prediction']
        logger.info(f"Prediction complete: {hemoglobin:.2f} g/dL")
        
        # DETERMINE ANEMIA STATUS (WHO GUIDELINES)
        is_pregnant = (pregnancy_status == "Yes")
        anemia_status = determine_anemia_status(
            hemoglobin_gdl=hemoglobin,
            age=int(age),
            sex=gender,
            is_pregnant=is_pregnant,
            trimester=2  # Default to 2nd trimester if pregnant
        )
        
        logger.info(f"Anemia status: {anemia_status}")
        
        # SAVE RESULTS TO REALTIME DATABASE /result
        # Use PH standard time
        philippine_tz = pytz.timezone('Asia/Manila')
        timestamp_iso = datetime.now(philippine_tz).strftime('%Y-%m-%dT%H:%M:%S')
        
        result_ref = db.reference('result')
        new_result = result_ref.push({
            'Patient_id': patient_id,  # Use Firebase key (not generated code)
            'Data_id': record_id,  # Add Data_id reference
            'Hemoglobin_level': f"{hemoglobin:.1f}",
            'Anemia_status': anemia_status,  # Changed from 'Anaemia_status'
            'Heart_rate': heart_rate,
            'Blood_oxygen': str(blood_oxygen),  # Changed from 'SpO2'
            'Timestamp_result': timestamp_iso  # Changed to ISO 8601 format
        })
        
        result_id = new_result.key
        logger.info(f"Results saved to /result/{result_id}")
        
        # UPDATE LATEST_DATA_ID IN FIRESTORE PATIENT DOC
        firestore_client.collection('patient').document(patient_id).update({
            'Latest_data_id': record_id
        })
        
        logger.info("=== Prediction complete ===")
        
        # RETURN RESULTS
        return {
            'success': True,
            'result_id': result_id,
            'hemoglobin_prediction': hemoglobin,
            'anemia_status': anemia_status,
            'eye_hemoglobin': result['eye_hemoglobin'],
            'ppg_hemoglobin': result['ppg_hemoglobin'],
            'weights': result['weights'],
            'segmentation_stats': result['segmentation_stats']
        }
        
    except https_fn.HttpsError:
        raise
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        logger.error(traceback.format_exc())
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INTERNAL,
            message=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8080))
    
    logger.info(f"Starting server on port {port}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )