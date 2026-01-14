import gradio as gr
from nivra_agent import nivra_chat

demo = gr.ChatInterface(
    nivra_chat,
    title="🩺 Nivra AI Healthcare Assistant",
    description="India-first symptom diagnosis: ClinicalBERT + Medical RAG",
    examples=[
        "I have fever and chills",
        "Skin rash and itching", 
        "Stomach pain, vomiting"
    ]
)

if __name__ == "__main__":
    demo.queue().launch()
