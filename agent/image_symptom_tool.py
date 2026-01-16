from langchain_core.tools import tool
import requests
import base64
from PIL import Image
import io

@tool
def analyze_symptom_image(image_url: str, image_description: str = "") -> str:
    """
    Analyzes symptom images using vision classification model via FastAPI backend.
    Input: Image URL and optional image description
    Output: Image symptom analysis with confidence scores 
    """
    try:
        # Download image from URL
        print(f"📥 Downloading image from: {image_url}")
        image_response = requests.get(image_url, timeout=120)
        image_response.raise_for_status()
        image = Image.open(io.BytesIO(image_response.content)).convert('RGB')
        
        # Convert to base64 for FastAPI transmission
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Call your HF FastAPI Space
        api_url = "https://datdevsteve-nivra-vision-diagnosis.hf.space/run/predict"
        payload = {
            "data": [img_base64],
            "description": image_description or ""
        }
        
        print("🔬 Calling Nivra Vision FastAPI backend...")
        response = requests.post(api_url, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        # Extract diagnosis from HF Space response format
        if "data" in result and len(result["data"]) > 0:
            diagnosis = result["data"][0]
            return f"""
[SYMPTOM IMAGE ANALYSIS - SUCCESS]:
✅ FastAPI Backend Response: {diagnosis}

🔍 **Image Description**: {image_description or "Not provided"}
📡 **Backend**: nivra-vision-diagnosis HF Space"""
        else:
            return "[SYMPTOM IMAGE ANALYSIS - WARNING]: No diagnosis returned from backend"
            
    except requests.exceptions.Timeout:
        return "[SYMPTOM IMAGE ANALYSIS - ERROR]: Backend timeout. Please try again."
    except requests.exceptions.RequestException as e:
        return f"[SYMPTOM IMAGE ANALYSIS - ERROR]: Network error: {str(e)}"
    except Exception as e:
        return f"[SYMPTOM IMAGE ANALYSIS - ERROR]: Unexpected error: {str(e)}"
