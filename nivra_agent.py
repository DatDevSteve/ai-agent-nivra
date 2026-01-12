#=========================================
#|| NIVRA AI HEALTHCARE ASSISTANT AGENT ||
#=========================================

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_react_agent  # ✅ Fixed import
from langchain_experimental.agents import create_tool_calling_agent  # ✅ New location
from langchain_core.tools import tool
from agent.rag_retriever import NivraRAGRetriever
from agent.text_symptom_tool import analyze_symptom_text
from agent.image_symptom_tool import analyze_symptom_image
from dotenv import load_dotenv
import os

load_dotenv()

# ✅ Fix: Proper RAG tool instantiation
rag = NivraRAGRetriever()
rag_tool = rag.getRelevantDocs  # Method reference (not class method)

# Instantiate LLM
llm = ChatGroq(
    temperature=0.1,
    model_name="llama-3.1-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

tools = [analyze_symptom_image, analyze_symptom_text, rag_tool]

system_prompt = """[Your exact system prompt - keep as-is]"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# ✅ Fixed: Correct LangChain v0.3+ imports
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)

def nivra_chat(user_input: str, chat_history: list = []):
    """Main chat function to be invoked via mobile app"""
    response = agent_executor.invoke({
        "input": user_input,
        "chat_history": chat_history
    })
    return response["output"]
