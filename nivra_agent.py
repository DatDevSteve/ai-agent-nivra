#=========================================
#|| NIVRA AI HEALTHCARE ASSISTANT ||
#=========================================

from langchain_groq import ChatGroq
from agent.rag_retriever import NivraRAGRetriever
from agent.text_symptom_tool import analyze_symptom_text
from agent.image_symptom_tool import analyze_symptom_image
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize tools
rag = NivraRAGRetriever()
llm = ChatGroq(
    temperature=0.1,
    model_name="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# ✅ YOUR EXACT SYSTEM PROMPT (preserved perfectly)
SYSTEM_PROMPT = """You are Nivra, a smart and helpful AI Healthcare Assistant with multimodal capabilities. Your role is to only provide medical attention to the user in a structured format of text or voice.

You are equipped with these tools:
- analyze_symptom_image: Performs symptom analysis on provided image input to diagnose skin conditions
- analyze_symptom_text: Performs symptom analysis on provided text input to diagnose diseases  
- rag_tool: Refers to medical knowledgebase for additional context

Always remember these rules:
- Output text should be structured for easy TTS narration when [VOICE MODE ONLY] is detected
- You ARE NOT A DOCTOR - provide disease information ONLY, NO treatment advice  
- Always provide doctor referral for Cancer, Dengue, Malaria, or high-risk diagnoses
- ALWAYS correlate with rag_tool for medical knowledge
- Keep descriptions short/crisp to limit token usage
- Always remember to use the tokens for every output component, such as [TOOLS USED][/TOOLS USED], [SYMPTOMS][/SYMPTOMS] etc as given in example

Basic Output Structure:
[TOOLS USED]
[SYMPTOMS] 
[PRIMARY DIAGNOSIS]
[DIAGNOSIS DESCRIPTION with RAG Knowledgebase]
[FIRST AID] 
[EMERGENCY CONSULTATION REQUIRED]

FEW-SHOT EXAMPLES (FOLLOW EXACT FORMAT):

(EXAMPLE 1- Voice Input- with [VOICE MODE] Token present in input)
Input: [VOICE MODE] "I have fever, chills and severe headache."
---
[TOOLS USED] analyze_symptom_text, rag_tool [/TOOLS USED]
[SYMPTOMS] Fever, Chills, Headache [/SYMPTOMS]
[PRIMARY DIAGNOSIS] Malaria (78% confidence) [/PRIMARY DIAGNOSIS] 
[DIAGNOSIS DESCRIPTION]
Malaria is caused by Plasmodium parasite spread by Anopheles mosquitoes. 
It multiplies in red blood cells causing fever, chills, headache cycles.
Common in India during monsoon season.
[/DIAGNOSIS DESCRIPTION]
[FIRST AID]
Rest completely and drink plenty of fluids. Seek immediate medical attention for malaria test and treatment.
[/FIRST AID]
[EMERGENCY] Yes [/EMERGENCY]

(EXAMPLE 2- Image Input)
Input: "I have skin rash" + rash.jpg
---
[TOOLS USED] analyze_symptom_image, rag_tool [/TOOLS USED]
[SYMPTOMS] Red scaly patches, itching [/SYMPTOMS]
[PRIMARY DIAGNOSIS] Psoriasis (82% confidence) [/PRIMARY DIAGNOSIS]
[DIAGNOSIS DESCRIPTION]
Psoriasis is a chronic autoimmune skin disorder causing rapid skin cell growth. 
Results in thick, dry, scaly patches. Non-contagious, affects 2-3% population.
[/DIAGNOSIS DESCRIPTION]
[FIRST AID]
Keep skin clean and moisturized. Avoid scratching. Consult dermatologist for management.
[/FIRST AID]
[EMERGENCY] No [/EMERGENCY]

(EXAMPLE 3- Low Confidence)
Input: "Stomach pain, vomiting frequently"
---
[TOOLS USED] analyze_symptom_text, rag_tool [/TOOLS USED]
[SYMPTOMS] Stomach pain, vomiting [/SYMPTOMS]
[PRIMARY DIAGNOSIS] Gastritis or Gastroenteritis [/PRIMARY DIAGNOSIS]
[DIAGNOSIS DESCRIPTION]
Multiple causes possible: acidity, infection, food poisoning. Clinical evaluation required.
[/DIAGNOSIS DESCRIPTION]
[FIRST AID]
Seek medical consultation immediately. Ultrasound may be needed.
[/FIRST AID]
[EMERGENCY] Yes [/EMERGENCY]

**CRITICAL**: High-risk triggers (EMERGENCY=Yes): melanoma, basal_cell_carcinoma, dengue, malaria, typhoid, cancer"""

def nivra_chat(user_input, chat_history=None):
    """DEBUG VERSION - Shows EXACT error"""
    
    # Input handling
    if isinstance(user_input, dict):
        user_input = user_input.get('text', '') or user_input.get('message', '')
    user_input = str(user_input).strip()
    
    print(f"🔍 DEBUG: Input received: '{user_input}'")
    
    input_lower = user_input.lower()
    text_keywords = ['fever', 'headache', 'cough', 'pain', 'vomiting', 'chills']
    
    tools_used = []
    tool_results = []
    
    # TEST TEXT TOOL FIRST
    if any(keyword in input_lower for keyword in text_keywords):
        print("🧪 TESTING analyze_symptom_text...")
        try:
            print("📡 Calling HF Space: https://datdevsteve-nivra-text-diagnosis.hf.space")
            symptom_result = analyze_symptom_text.invoke(user_input)
            print(f"✅ TEXT TOOL SUCCESS: {symptom_result[:100]}...")
            tools_used.append("analyze_symptom_text")
            tool_results.append(symptom_result)
        except Exception as e:
            error_msg = f"TEXT TOOL FAILED: {str(e)}"
            print(f"❌ {error_msg}")
            tool_results.append(error_msg)
    
    # TEST RAG
    print("🧪 TESTING RAG...")
    try:
        rag_result = rag.getRelevantDocs(user_input)
        print(f"✅ RAG SUCCESS: {str(rag_result)[:100]}...")
        tools_used.append("rag_tool")
        tool_results.append(rag_result)
    except Exception as e:
        error_msg = f"RAG FAILED: {str(e)}"
        print(f"❌ {error_msg}")
        tool_results.append(error_msg)
    
    # Convert to strings
    tool_results_str = [str(r) for r in tool_results]
    tool_results_text = "\n".join(tool_results_str)
    
    # Quick fallback if tools fail
    if "FAILED" in tool_results_text:
        return f"""[TOOLS USED] Tools failed - Network issue
[SYMPTOMS] {user_input}
[PRIMARY DIAGNOSIS] Possible viral fever/infection
[DIAGNOSIS DESCRIPTION] Fever+chills suggests infection. ClinicalBERT backend temporarily unavailable.
[FIRST AID] Rest, hydrate, paracetamol. Monitor temperature.
[EMERGENCY] No - but consult doctor if >3 days"""

    # Your normal flow
    final_prompt = f"""{SYSTEM_PROMPT}

TOOL RESULTS:
{tool_results_text}
q 
USER INPUT: {user_input}

Provide diagnosis:"""
    
    try:
        response = llm.invoke(final_prompt)
        return response.content.strip()
    except Exception as e:
        return f"LLM FAILED: {str(e)}"
