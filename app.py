import gradio as gr
from nivra_agent import nivra_chat

demo = gr.ChatInterface(
    nivra_chat,
    textbox=gr.Textbox(),
    fill_height=True,
    title="🩺 Nivra AI Agent",
    description="Space to access Nivra's Agentic Interface",
    examples=[
        "I have fever and chills",
        "Patient presents Skin rash and itching", 
        "Patient presents Stomach pain and vomiting"
    ]
)

if __name__ == "__main__":
    demo.queue().launch(theme=gr.themes.Ocean())
