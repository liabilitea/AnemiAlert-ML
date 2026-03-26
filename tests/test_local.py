# Test script for local Docker container.
# Run this after starting your container with: docker run -p 8080:8080 hb-prediction

import requests
import json
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8080"
SAMPLE_IMAGE = "tests/sample_data/eye_sample.jpg"  # You need to add this file

# Sample test data
TEST_DATA = {
    "ir_value": 123456.78,
    "red_value": 98765.43,
    "age": 25,
    "gender": "Male"
}


def test_health():
    # Test health endpoint.
    print("\n" + "="*50)
    print("Testing /health endpoint")
    print("="*50)
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("Health check passed")
            return True
        else:
            print("Health check failed")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_predict():
    # Test prediction endpoint.
    print("\n" + "="*50)
    print("Testing /predict endpoint")
    print("="*50)
    
    # Check if sample image exists
    if not Path(SAMPLE_IMAGE).exists():
        print(f"Sample image not found: {SAMPLE_IMAGE}")
        print("Please add a sample eye image to test with.")
        return False
    
    try:
        # Prepare multipart form data
        with open(SAMPLE_IMAGE, 'rb') as f:
            files = {'eye_image': ('eye.jpg', f, 'image/jpeg')}
            data = TEST_DATA
            
            print(f"\nSending request with:")
            print(f"   - Image: {SAMPLE_IMAGE}")
            print(f"   - IR Value: {data['ir_value']}")
            print(f"   - Red Value: {data['red_value']}")
            print(f"   - Age: {data['age']}")
            print(f"   - Gender: {data['gender']}")
            
            response = requests.post(
                f"{BASE_URL}/predict",
                files=files,
                data=data,
                timeout=60
            )
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("\nPrediction successful!")
                print(f"   Final Hb: {result['hemoglobin_prediction']} g/dL")
                print(f"   Eye Hb: {result['eye_hemoglobin']} g/dL")
                print(f"   PPG Hb: {result['ppg_hemoglobin']} g/dL")
                return True
            else:
                print(f"Prediction failed: {result.get('error')}")
                return False
        else:
            print("Request failed")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    # Run all tests.
    print("\n" + "="*50)
    print("LOCAL DOCKER CONTAINER TESTING")
    print("="*50)
    print(f"Target: {BASE_URL}")
    print("\nMake sure your container is running:")
    print("   docker run -p 8080:8080 hb-prediction")
    
    # Test health first
    health_ok = test_health()
    
    if not health_ok:
        print("\nHealth check failed. Container may not be running.")
        print("   Start it with: docker run -p 8080:8080 hb-prediction")
        return
    
    # Test prediction
    predict_ok = test_predict()
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Health Check: {'PASS' if health_ok else 'FAIL'}")
    print(f"Prediction:   {'PASS' if predict_ok else 'FAIL'}")
    
    if health_ok and predict_ok:
        print("\nAll tests passed! Container is working correctly.")
    else:
        print("\nSome tests failed. Check the logs above.")


if __name__ == "__main__":
    main()