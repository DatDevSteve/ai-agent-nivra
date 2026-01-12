from langchain_core.tools import tool
import json
import os
import requests
from PIL import Image
import io

import torch

@tool
def analyze_symptom_image(image_url:str, image_description: str= "") -> str:
    """
    Analyzes symptom mages using vision classification model.
    Input: Image URL and optional image description
    Output: Image symptom analysis with confidence scores 
    """
    try:
        #load image labels json
        try:
            with open("labels.json", "r", encoding="utf-8") as f:
                label_config = json.load(f)
            
            # Extract class names from your structured format
            if "class_names" in label_config:
                image_labels = label_config["class_names"]
            elif "id2label" in label_config:
                # Sort by ID to maintain order
                image_labels = [label_config["id2label"][str(i)] 
                               for i in range(label_config["num_labels"])]
            else:
                image_labels = list(label_config.values())
                
            num_labels = label_config.get("num_labels", len(image_labels))
            image_size = label_config.get("image_size", {"height": 224, "width": 224})
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            return "[!] FATAL ERROR OCCURED WHILE LOADING TOOL: {e}"
        
        #fetch image
        image_response = requests.get(image_url, timeout=10)
        image = Image.open(io.BytesIO(image_response.content)).convert('RGB')

        #load model
        from transformers import AutoImageProcessor, AutoModel
        model_name = "datdevsteve/dinov2-nivra-finetuned"
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        #process image
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.nn.functional.softmax(logits, dim=-1)
        
        #fetch predictions
        top_idx = predictions.argmax().item()
        top_prediction = image_labels[top_idx]
        confidence = predictions.max().item()

        #fetch top 3 predictions
        top3_values, top3_indices = predictions.topk(3)
        top3_labels = [image_labels[idx.item] for idx in top3_indices[0]]
        top3_scores = top3_values[0].tolist()

        #create analysis
        analysis = f"""
[SYMPTOM IMAGE ANALYSIS- SUCCESS]:
 - Primary Analysis: {top_prediction.replace("_", " ").title()}: {confidence:.1%}
 - Top 3 Findings:
"""
        for i, (label, score) in enumerate(zip(top3_labels, top3_scores), 1):
            analysis += f"{i}. {label.replace("_", " ").title()}: {score:.1%}\n"
        analysis += f"\n Urgency: "
        if confidence > 0.8:
            analysis += "[HIGH] - It is suggested to consult doctor at the earliest for appropriate treatment."
        elif confidence > 0.6:
            analysis += "[MEDIUM] - Try asking for more information and correlate with symptoms."
        else:
            analysis += "[LOW] - Tool was unable to analyze and hence clinical review is suggested."
        return analysis.strip()
    except Exception as e:
        fallback = """
[SYMPTOM IMAGE ANALYSIS- ERROR]:
The tool was unable to analyze symptom images due to an inevitable error. Below is the error information:
[!]{e}
"""
        return fallback.format(image_url = image_url)