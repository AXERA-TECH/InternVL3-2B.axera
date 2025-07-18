import cv2
import gradio as gr
from llm import LLM
import llm as llm
import argparse
import socket

parser = argparse.ArgumentParser(description="Model configuration parameters")
parser.add_argument("--hf_model", type=str, default="./InternVL3-2B",
                    help="Path to HuggingFace model")
parser.add_argument("--axmodel_path", type=str, default="./InternVL3-2B_axmodel",
                    help="Path to save compiled axmodel of llama model")
parser.add_argument("--vit_model", type=str, default=None,
                    help="Path to save compiled axmodel of llama model")
args = parser.parse_args()

hf_model_path = args.hf_model
axmodel_path = args.axmodel_path
vit_axmodel_path = args.vit_model

gllm = LLM(hf_model_path, axmodel_path, vit_axmodel_path)

 # 获取本地 IP 地址
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def stop_generation():
    gllm.stop_generate()

def respond(prompt, video, image, is_image, video_segments,  image_segments_cols, image_segments_rows, history=None):
    if history is None:
        history = []
    if not prompt.strip():
        return history
    # append empty response to history
    
    gllm.tag = "video" if not is_image else "image"

    history.append((prompt, ""))
    yield history
    
    print(video)
    print(image)
    
    if is_image:
        img = cv2.imread(image)
        images_list = []
        if image_segments_cols == 1 and image_segments_rows == 1:
            images_list.append(img)
        elif image_segments_cols * image_segments_rows > 8:
            # gr.Error("image segments cols * image segments rows > 8")
            history[-1] = (prompt, history[-1][1] + "image segments cols * image segments rows > 8")
            yield history
            return
        else:
            height, width, _ = img.shape
            segment_width = width // image_segments_cols
            segment_height = height // image_segments_rows
            for i in range(image_segments_rows):
                for j in range(image_segments_cols):
                    x1 = j * segment_width
                    y1 = i * segment_height
                    x2 = (j + 1) * segment_width
                    y2 = (i + 1) * segment_height
                    segment = img[y1:y2, x1:x2]
                    images_list.append(segment)
    else:
        images_list = llm.load_video(video, num_segments = video_segments)
        
    for msg in gllm.generate(images_list, prompt):
        print(msg, end="", flush=True)
        history[-1] = (prompt, history[-1][1] + msg)
        yield history
    print("\n\n\n")
    
def chat_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## AXERA ChatBot\nUpload an image or video and chat with the llm!")
        with gr.Row():
            with gr.Column(scale=1):
                video = gr.Video(label="Upload Video", format="mp4")
                video_segments = gr.Slider(minimum=2, maximum=8, step=1, value=4, label="video segments")
                image = gr.Image(label="Upload Image", type="filepath")
                image_segments_cols = gr.Slider(minimum=1, maximum=4, step=1, value=1, label="image cols segments")
                image_segments_rows = gr.Slider(minimum=1, maximum=4, step=1, value=1, label="image rows segments")
                checkbox = gr.Checkbox(label="Use Image")
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=650)
                prompt = gr.Textbox(placeholder="Type your message...", label="Prompt", value="描述一下这组图片")
                with gr.Row():
                    btn_chat = gr.Button("Chat", variant="primary")
                    btn_stop = gr.Button("Stop", variant="stop")

            btn_stop.click(fn=stop_generation, inputs=None, outputs=None)
            
            btn_chat.click(
                fn=respond,
                inputs=[prompt, video, image, checkbox, video_segments, image_segments_cols, image_segments_rows, chatbot],
                outputs=chatbot
            )
            
            def on_video_uploaded(video):
                if video is not None:
                    return gr.update(value=False)
                return gr.update()

            def on_image_uploaded(image):
                if image is not None:
                    return gr.update(value=True)
                return gr.update()

            video.change(fn=on_video_uploaded, inputs=video, outputs=checkbox)
            image.change(fn=on_image_uploaded, inputs=image, outputs=checkbox)
       

        local_ip = get_local_ip()
        server_port = 7860
        print(f"HTTP 服务地址: http://{local_ip}:{server_port}")
        demo.launch(server_name=local_ip, server_port=server_port)

if __name__ == "__main__":
    chat_interface()