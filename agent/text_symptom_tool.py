import json
import requests
from langchain_core.tools import tool

@tool
def analyze_symptom_text(symptoms: str) -> str:
    """
    Analyzes text-based symptoms using classification model via api backend.
    Input: Patient's symptom description text.
    Output: Predicted conditions with confidence scores.
    """
    try:
        print(f"🩺 Analyzing symptoms: {symptoms[:50]}...")
        
        # Call your HF ClinicalBERT FastAPI Space
        api_url = "https://datdevsteve-nivra-text-diagnosis.hf.space/run/predict"
        payload = {
            "data": [symptoms],
            "fn_index": 0  # Default prediction function
        }
        
        print("🔬 Calling ClinicalBERT FastAPI backend...")
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        # Extract diagnosis from HF Space response format
        if "data" in result and len(result["data"]) > 0:
            diagnosis = result["data"][0]
            
            # Parse confidence if available, else default format
            if isinstance(diagnosis, list) and len(diagnosis) > 0:
                diagnosis = diagnosis[0]
                
            return f"""
[TEXT SYMPTOM ANALYSIS - SUCCESS]:
✅ FastAPI Backend Response: {diagnosis}

📡 **Backend**: nivra-text-diagnosis HF Space"""
        else:
            # Fallback with generic advice
            return "[TEXT SYMPTOM ANALYSIS - WARNING]: No diagnosis returned from backend. Please consult a doctor."
            
    except requests.exceptions.Timeout:
        return "[TEXT SYMPTOM ANALYSIS - ERROR]: Analysis timeout. Please try again or consult a doctor."
    except requests.exceptions.RequestException as e:
        return f"[TEXT SYMPTOM ANALYSIS - ERROR]: Network error: {str(e)}. Please consult a doctor."
    except Exception as e:
        return f"[TEXT SYMPTOM ANALYSIS - ERROR]: Unexpected error: {str(e)}. Please consult a doctor."
