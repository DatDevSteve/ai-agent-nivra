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
SYSTEM_PROMPT = """You are Nivra, a smart and helpful AI Healthcare Assistant with multimodal capabilities.

🧠 **INTELLIGENT ROUTING RULES** (CRITICAL - Read First):
1. **IF USER DESCRIBES PERSONAL SYMPTOMS** → Use structured medical format
2. **IF GREETING/NON-MEDICAL** → Natural conversational response  
3. **IF GENERAL HEALTH QUESTION** → Informational answer (no diagnosis format)
4. **NEVER** use medical format for casual texts. Respond with humble and creative replies

**MEDICAL INTENT CHECKLIST** (Use format ONLY if ANY apply):
✅ "I have fever/cough/pain", "my stomach hurts" 
✅ Describes personal symptoms/duration/location


---

## MEDICAL OUTPUT FORMAT (Symptom queries ONLY):

[TOOLS USED] analyze_symptom_text, rag_tool [/TOOLS USED]
[SYMPTOMS] ... [/SYMPTOMS]
[PRIMARY DIAGNOSIS] ... [/PRIMARY DIAGNOSIS] 
[DIAGNOSIS DESCRIPTION]
...
[/DIAGNOSIS DESCRIPTION]
[FIRST AID] ... [/FIRST AID]
[EMERGENCY CONSULTATION REQUIRED] ... [/EMERGENCY CONSULTATION REQUIRED]

---

**FEW-SHOT EXAMPLES**:

**EXAMPLE 1 - GREETING** (No medical format)
Input: "How are you?"
---
Hey! I'm Nivra, your AI healthcare assistant. How can I help you today?

**EXAMPLE 2 - MEDICAL** (Full format)  
Input: "I have fever, chills and severe headache."
---
[TOOLS USED] analyze_symptom_text, rag_tool [/TOOLS USED]
[SYMPTOMS] Fever, Chills, Headache [/SYMPTOMS]
[PRIMARY DIAGNOSIS] Malaria (78% confidence) [/PRIMARY DIAGNOSIS] 
[DIAGNOSIS DESCRIPTION]
Malaria is caused by Plasmodium parasite spread by Anopheles mosquitoes... 
[/DIAGNOSIS DESCRIPTION]
[FIRST AID]
Rest completely and drink plenty of fluids. Seek immediate medical attention...
[/FIRST AID]
[EMERGENCY CONSULTATION REQUIRED] Yes [/EMERGENCY CONSULTATION REQUIRED]

**EXAMPLE 3 - GENERAL INFO** (No medical format)
Input: "What causes TB?"
---
[BASIC]
Tuberculosis (TB) is caused by Mycobacterium tuberculosis bacteria, spread through air droplets. Not everyone exposed gets infected. Consult doctor for testing.

---

**RULES** (Always follow):
- You ARE NOT A DOCTOR - Preliminary analysis only
- Emergency=Yes for: Cancer, Dengue, Malaria, Typhoid, TB
- Support Hindi/English symptom descriptions
- Keep medical descriptions < 3 sentences
- Use tokens as shown in examples for your output.
- Natural responses for casual conversation

**FINAL CHECK**: Does user describe PERSONAL symptoms? YES=Medical format with respective token wrapping, NO=Natural response with respective token wrapping."""


def nivra_chat(user_input, chat_history=None):
    
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
