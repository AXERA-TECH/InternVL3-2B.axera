import mimetypes
import os
import gradio as gr
import requests
import json, time
from llm import LLM

hf_model_path = "../InternVL3-2B"
axmodel_path = "../InternVL3-2B_axmodel/"
vit_axmodel_path = "../vit_axmodel/internvl3_2b_vit_slim.axmodel"
llm = LLM(hf_model_path, axmodel_path, vit_axmodel_path)

def stop_generation():
    llm.stop_generate()

def respond(prompt, video, num_segments, history=None):
    if history is None:
        history = []
    if not prompt.strip():
        return history
    # append empty response to history
   
    
    history.append((prompt, ""))
    yield history
    print(video)
    for msg in llm.generate(video, prompt, num_segments):
        print(msg, end="", flush=True)
        history[-1] = (prompt, history[-1][1] + msg)
        yield history
    print("\n\n\n")
    
def chat_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## Chat with LLM\nUpload an image and chat with the model!")
        with gr.Row():
            with gr.Column(scale=1):
                video = gr.Video(label="Upload Video", format="mp4")
                num_segments = gr.Slider(minimum=2, maximum=8, step=1, value=4, label="num_segments")
            with gr.Column(scale=3):
                chatbot = gr.Chatbot()
                prompt = gr.Textbox(placeholder="Type your message...", label="Prompt", value="描述一下这个视频")
                with gr.Row():
                    btn_chat = gr.Button("Chat", variant="primary")
                    btn_stop = gr.Button("Stop", variant="stop")

            btn_stop.click(fn=stop_generation, inputs=None, outputs=None)
            btn_chat.click(
                fn=respond,
                inputs=[prompt, video, num_segments, chatbot],
                outputs=chatbot
            )

        demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    chat_interface()