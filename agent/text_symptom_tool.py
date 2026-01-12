import json
from langchain_core.tools import tool
from transformers import AutoTokenzier, AutoModelForSequenceClassification
import torch

@tool
def analyze_symptom_text(symptoms: str) -> str:
    """
    Analyzes text based symptoms using Text Classification Model.
    Input: Patient's symptom description text.
    Output: Predicted conditions with confidence scores. 
    """
    #Load model:
    model_name = "datdevsteve/nivra-text-diagnosis"
    tokenizer = AutoTokenzier.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    inputs = tokenizer(symptoms, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    #Map Predictions:

    try:
        with open("labels.json", "r", encoding="utf-8") as f:
            disease_labels = json.load(f)
        
        # Ensure labels is a list (handles both list and dict formats)
        if isinstance(disease_labels, dict):
            disease_labels = list(disease_labels.values())
        elif not isinstance(disease_labels, list):
            disease_labels = list(disease_labels)
            
    except FileNotFoundError:
        # Fallback labels if JSON not found
        disease_labels = ["fever_malaria", "dengue", "covid", "typhoid", "pneumonia", "tb"]
        print("Warning: labels.json not found, using fallback labels")
    except json.JSONDecodeError:
        # Fallback if JSON is malformed
        disease_labels = ["fever_malaria", "dengue", "covid", "typhoid", "pneumonia", "tb"]
        print("Warning: labels.json malformed, using fallback labels")
    
    top_prediction = disease_labels[predictions.argmax().item()]
    confidence = predictions.max().item()

    return f"Primary diagnosis: {top_prediction.replace("_", " ").title()}(Confidence: {confidence:.2%} )"