"""
Preprocessing functions for hemoglobin prediction models.
Matches the exact preprocessing used during training.
"""

import cv2
import numpy as np
from typing import Dict, Tuple
from PIL import Image


def preprocess_unet_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for UNet++ segmentation model.
    
    Steps:
    1. Convert BGR to RGB (if needed)
    2. Resize to 256x256
    3. Normalize to [0, 1]
    4. Add batch dimension
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Preprocessed image ready for UNet++ (1, 256, 256, 3)
    """
    # Ensure RGB format
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Resize to model input size
    image = cv2.resize(image, (256, 256))
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image


def extract_colored_segmentation(
    original_image: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Extract colored conjunctiva region from original image using segmentation mask.
    Background becomes transparent.
    
    Args:
        original_image: Original eye image (H, W, 3) in RGB
        mask: Segmentation mask (H, W, 1) with values [0, 1]
        threshold: Threshold for binarizing mask
        
    Returns:
        RGBA image with colored conjunctiva and transparent background (H, W, 4)
    """
    # Remove batch dimension if present
    if len(mask.shape) == 4:
        mask = mask[0]
    if len(mask.shape) == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    
    # Binarize mask
    binary_mask = (mask > threshold).astype(np.uint8)
    
    # Resize mask to match original image size
    if binary_mask.shape != original_image.shape[:2]:
        binary_mask = cv2.resize(
            binary_mask,
            (original_image.shape[1], original_image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    
    # Create RGBA image
    rgba_image = np.zeros(
        (original_image.shape[0], original_image.shape[1], 4),
        dtype=np.uint8
    )
    
    # Copy RGB channels
    rgba_image[:, :, :3] = original_image
    
    # Set alpha channel based on mask (255 = opaque, 0 = transparent)
    rgba_image[:, :, 3] = binary_mask * 255
    
    return rgba_image


def extract_lab_features(image: np.ndarray, mask: np.ndarray = None) -> Dict[str, float]:
    """
    Extract 27 statistical features from LAB color space.
    Only analyzes pixels within the mask (conjunctiva region).
    
    Args:
        image: RGB image (H, W, 3)
        mask: Optional binary mask (H, W) - only analyze masked pixels
        
    Returns:
        Dictionary with 27 LAB features:
        - L_mean, L_std, L_median, L_min, L_max, L_25, L_75, L_var, L_range
        - a_mean, a_std, a_median, a_min, a_max, a_25, a_75, a_var, a_range
        - b_mean, b_std, b_median, b_min, b_max, b_25, b_75, b_var, b_range
    """
    # Resize to 224x224 (as in training)
    image_resized = cv2.resize(image, (224, 224))
    
    # Convert RGB to LAB
    lab_image = cv2.cvtColor(image_resized, cv2.COLOR_RGB2LAB)
    
    # If mask provided, resize it and apply
    if mask is not None:
        mask_resized = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        mask_bool = mask_resized > 0
        
        # Extract only masked pixels for each channel
        l_channel = lab_image[:, :, 0][mask_bool]
        a_channel = lab_image[:, :, 1][mask_bool]
        b_channel = lab_image[:, :, 2][mask_bool]
    else:
        # Use all pixels
        l_channel = lab_image[:, :, 0].flatten()
        a_channel = lab_image[:, :, 1].flatten()
        b_channel = lab_image[:, :, 2].flatten()
    
    # Calculate statistics for each channel
    def calculate_stats(channel: np.ndarray, prefix: str) -> Dict[str, float]:
        return {
            f'{prefix}_mean': float(np.mean(channel)),
            f'{prefix}_std': float(np.std(channel)),
            f'{prefix}_median': float(np.median(channel)),
            f'{prefix}_min': float(np.min(channel)),
            f'{prefix}_max': float(np.max(channel)),
            f'{prefix}_p25': float(np.percentile(channel, 25)),
            f'{prefix}_p75': float(np.percentile(channel, 75)),
            f'{prefix}_var': float(np.var(channel)),
            f'{prefix}_range': float(np.max(channel) - np.min(channel))
        }
    
    features = {}
    features.update(calculate_stats(l_channel, 'L'))
    features.update(calculate_stats(a_channel, 'a'))
    features.update(calculate_stats(b_channel, 'b'))
    
    return features


def prepare_eye_features(
    lab_features: Dict[str, float],
    age: int,
    gender: str
) -> Dict[str, float]:
    """
    Prepare feature dictionary for TFDF eye regression model.
    
    Args:
        lab_features: 27 LAB statistical features
        age: Patient age
        gender: Patient gender ("Male" or "Female")
        
    Returns:
        Dictionary with 29 features ready for TFDF model (lowercase keys)
    """
    # Encode gender (0 = Male, 1 = Female)
    gender_encoded = 1 if gender.lower() == "female" else 0
    
    # Convert feature names to lowercase to match model signature
    features = {key.lower(): value for key, value in lab_features.items()}
    features['age'] = float(age)
    features['gender'] = float(gender_encoded)
    
    return features


def prepare_ppg_features(
    ir_value: float,
    red_value: float,
    age: int,
    gender: str
) -> Dict[str, any]:
    """
    Prepare feature dictionary for TFDF PPG regression model.
    
    Args:
        ir_value: Infrared sensor value
        red_value: Red light sensor value
        age: Patient age
        gender: Patient gender ("Male" or "Female")
        
    Returns:
        Dictionary with 4 features ready for TFDF model
    """
    # TFDF PPG model expects sex as string ('0' or '1')
    sex_encoded = '1' if gender.lower() == "female" else '0'
    
    return {
        'ir_value': float(ir_value),
        'red_value': float(red_value),
        'age': int(age),
        'sex': sex_encoded
    }


def validate_image(image: np.ndarray) -> Tuple[bool, str]:
    """
    Validate that image meets requirements.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if image is None:
        return False, "Image is None"
    
    if len(image.shape) not in [2, 3]:
        return False, f"Invalid image shape: {image.shape}"
    
    if len(image.shape) == 3 and image.shape[2] not in [3, 4]:
        return False, f"Invalid number of channels: {image.shape[2]}"
    
    # Check size (at least 64x64, max 4096x4096)
    h, w = image.shape[:2]
    if h < 64 or w < 64:
        return False, f"Image too small: {w}x{h}"
    if h > 4096 or w > 4096:
        return False, f"Image too large: {w}x{h}"
    
    return True, ""


def validate_ppg_values(ir_value: float, red_value: float) -> Tuple[bool, str]:
    """
    Validate PPG sensor values.
    
    Args:
        ir_value: Infrared sensor value
        red_value: Red light sensor value
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if ir_value <= 0:
        return False, "ir_value must be positive"
    if red_value <= 0:
        return False, "red_value must be positive"
    
    # Check for reasonable ranges (adjust based on your sensor)
    if ir_value > 1e9 or red_value > 1e9:
        return False, "PPG values unreasonably large"
    
    return True, ""