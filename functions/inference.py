"""
Model loading and inference pipeline for hemoglobin prediction.
"""

import os
import logging
import numpy as np
import tensorflow as tf
import tensorflow_decision_forests as tfdf  # Must import to register ops
from typing import Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
import cv2

from preprocessing import (
    preprocess_unet_image,
    extract_colored_segmentation,
    extract_lab_features,
    prepare_eye_features,
    prepare_ppg_features
)

logger = logging.getLogger(__name__)


class HemoglobinPredictor:
    """
    Loads and manages all three models for hemoglobin prediction.
    """
    
    def __init__(self, models_dir: str = "./models"):
        """
        Initialize predictor and load all models.
        
        Args:
            models_dir: Directory containing model files
        """
        self.models_dir = models_dir
        self.unet_model = None
        self.eye_model = None
        self.ppg_model = None
        
        logger.info("Loading models...")
        self._load_models()
        logger.info("All models loaded successfully")
    
    def _load_models(self):
        """Load all three models into memory."""
        
        # Load UNet++ Segmentation Model (.h5 file)
        unet_path = os.path.join(self.models_dir, "unet_segmentation.h5")
        if not os.path.exists(unet_path):
            raise FileNotFoundError(f"UNet model not found at {unet_path}")
        
        logger.info(f"Loading UNet++ from {unet_path}")
        self.unet_model = tf.keras.models.load_model(unet_path, compile=False)
        logger.info(f"UNet++ loaded: input shape {self.unet_model.input_shape}")
        
        # Load TFDF Eye Regression Model (SavedModel folder)
        eye_path = os.path.join(self.models_dir, "eye_regression")
        if not os.path.exists(eye_path):
            raise FileNotFoundError(f"Eye regression model not found at {eye_path}")
        
        logger.info(f"Loading TFDF Eye model from {eye_path}")
        self.eye_model = tf.saved_model.load(eye_path)
        logger.info("TFDF Eye model loaded")
        
        # Load TFDF PPG Regression Model (SavedModel folder)
        ppg_path = os.path.join(self.models_dir, "ppg_regression")
        if not os.path.exists(ppg_path):
            raise FileNotFoundError(f"PPG regression model not found at {ppg_path}")
        
        logger.info(f"Loading TFDF PPG model from {ppg_path}")
        self.ppg_model = tf.saved_model.load(ppg_path)
        logger.info("TFDF PPG model loaded")
    
    def predict_segmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Run UNet++ segmentation to find conjunctiva.
        
        Args:
            image: Preprocessed image (1, 256, 256, 3)
            
        Returns:
            Segmentation mask (256, 256, 1) with values [0, 1]
        """
        logger.info("Running UNet++ segmentation...")
        mask = self.unet_model.predict(image, verbose=0)
        logger.info(f"Segmentation complete: mask shape {mask.shape}")
        return mask
    
    def predict_eye_hb(self, features: Dict[str, float]) -> float:
        """
        Predict hemoglobin from eye features using TFDF model.
        
        Args:
            features: Dictionary with 29 features (27 LAB + Age + Gender)
            
        Returns:
            Predicted hemoglobin value
        """
        logger.info("Running TFDF Eye regression...")
        
        # Convert dict to DataFrame-like format expected by TFDF
        # IMPORTANT: Match exact casing expected by model signature
        # - Age, Gender, L_* parameters start with uppercase
        # - a_* and b_* parameters stay lowercase
        input_dict = {}
        for key, value in features.items():
            # Only capitalize if it starts with 'l', 'age', or 'gender'
            if key.startswith('l_'):
                # 'l_mean' -> 'L_mean'
                capitalized_key = 'L' + key[1:]
            elif key in ['age', 'gender']:
                # 'age' -> 'Age', 'gender' -> 'Gender'
                capitalized_key = key.capitalize()
            else:
                # 'a_mean', 'b_mean' etc stay lowercase
                capitalized_key = key
            input_dict[capitalized_key] = [value]
        
        # Predict
        prediction = self.eye_model.signatures["serving_default"](**input_dict)
        
        # Extract prediction value (TFDF returns a dict with tensor)
        # The output key might be 'output_1' or 'predictions' - check model
        output_key = list(prediction.keys())[0]
        hb_value = float(prediction[output_key].numpy()[0])
        
        logger.info(f"Eye Hb prediction: {hb_value:.2f}")
        return hb_value
    
    def predict_ppg_hb(self, features: Dict[str, any]) -> float:
        """
        Predict hemoglobin from PPG features using TFDF model.
        
        Args:
            features: Dictionary with 4 features (ir_value, red_value, age, sex)
            
        Returns:
            Predicted hemoglobin value
        """
        logger.info("Running TFDF PPG regression...")
        
        # Convert dict to format expected by TFDF
        input_dict = {key: [value] for key, value in features.items()}
        
        # Predict
        prediction = self.ppg_model.signatures["serving_default"](**input_dict)
        
        # Extract prediction value
        output_key = list(prediction.keys())[0]
        hb_value = float(prediction[output_key].numpy()[0])
        
        logger.info(f"PPG Hb prediction: {hb_value:.2f}")
        return hb_value
    
    def predict(
        self,
        image: np.ndarray,
        ir_value: float,
        red_value: float,
        age: int,
        gender: str
    ) -> Dict[str, any]:
        """
        Complete inference pipeline:
        1. Segment eye image
        2. Extract colored conjunctiva
        3. Extract LAB features from conjunctiva (parallel with step 4)
        4. Predict Hb from PPG (parallel with step 3)
        5. Combine predictions with weighted average (5% eye, 95% PPG)
        
        Args:
            image: Original eye image (RGB, any size)
            ir_value: Infrared sensor reading
            red_value: Red light sensor reading
            age: Patient age
            gender: Patient gender ("Male" or "Female")
            
        Returns:
            Dictionary with predictions and intermediate results
        """
        logger.info("Starting hemoglobin prediction pipeline")
        
        # Step 1: Preprocess and segment
        preprocessed_image = preprocess_unet_image(image)
        segmentation_mask = self.predict_segmentation(preprocessed_image)
        
        # Step 2: Extract colored conjunctiva with transparent background
        colored_segmentation = extract_colored_segmentation(
            image,
            segmentation_mask
        )
        
        # Extract just the RGB part for feature extraction (no alpha)
        conjunctiva_rgb = colored_segmentation[:, :, :3]
        
        # Extract binary mask for feature extraction
        binary_mask = (segmentation_mask[0, :, :, 0] > 0.5).astype(np.uint8)
        binary_mask_resized = cv2.resize(binary_mask, (image.shape[1], image.shape[0]))
        
        # Prepare features for both models
        lab_features = extract_lab_features(conjunctiva_rgb, binary_mask_resized)
        eye_features = prepare_eye_features(lab_features, age, gender)
        ppg_features = prepare_ppg_features(ir_value, red_value, age, gender)
        
        # Step 3 & 4: Run both regressions in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            eye_future = executor.submit(self.predict_eye_hb, eye_features)
            ppg_future = executor.submit(self.predict_ppg_hb, ppg_features)
            
            eye_hb = eye_future.result()
            ppg_hb = ppg_future.result()
        
        # Step 5: Weighted average (70% eye, 30% PPG)
        final_hb = (0.70 * eye_hb) + (0.30 * ppg_hb)
        
        logger.info(f"Final Hb prediction: {final_hb:.2f} g/dL")
        
        # Return comprehensive results
        return {
            "hemoglobin_prediction": round(final_hb, 2),
            "eye_hemoglobin": round(eye_hb, 2),
            "ppg_hemoglobin": round(ppg_hb, 2),
            "weights": {
                "eye": 0.70,
                "ppg": 0.30
            },
            "segmentation_stats": {
                "mask_coverage_percent": float(np.mean(binary_mask) * 100)
            }
        }


# Global predictor instance (loaded once on container startup)
predictor = None


def get_predictor() -> HemoglobinPredictor:
    """
    Get or create the global predictor instance.
    This ensures models are loaded only once.
    """
    global predictor
    if predictor is None:
        predictor = HemoglobinPredictor()
    return predictor