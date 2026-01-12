import gradio as gr
from nivra_agent import nivra_chat
import os

# Gradio interface for HF Spaces
def chat_interface(message, history, image=None, audio=None):
    # Build multimodal input
    input_text = message
    if image:
        input_text += f"\n[IMAGE: {image}]"
    if audio:
        input_text += f"\n[AUDIO: {audio}]"
    
    response = nivra_chat(input_text)
    history.append((message, response))
    return history, ""

demo = gr.ChatInterface(
    chat_interface,
    title="🩺 Nivra AI Healthcare Assistant",
    description="India-first symptom diagnosis: Text + Image + Voice",
    examples=[
        ["I have fever and chills"],
        ["Skin rash", gr.Image(type="filepath")],
        ["Stomach pain, vomiting"]
    ],
    multimodal=True
)

if __name__ == "__main__":
    demo.launch()
