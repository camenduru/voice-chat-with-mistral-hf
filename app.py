from __future__ import annotations

import os

import gradio as gr
import numpy as np
import torch
import torchaudio

DESCRIPTION = """# Speak with Llama2
TODO
"""

system_message = "\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
temperature = 0.9
top_p = 0.6
repetition_penalty = 1.2

import gradio as gr
import os
import time

import gradio as gr
from transformers import pipeline
import numpy as np

from gradio_client import Client

whisper_client = Client("https://sanchit-gandhi-whisper-large-v2.hf.space/")
text_client = Client("https://ysharma-explore-llamav2-with-tgi.hf.space/")

def transcribe(wav_path):
    
    return whisper_client.predict(
				wav_path,	# str (filepath or URL to file) in 'inputs' Audio component
				"transcribe",	# str in 'Task' Radio component
				api_name="/predict"
)
    

# Chatbot demo with multimodal input (text, markdown, LaTeX, code blocks, image, audio, & video). Plus shows support for streaming text.


def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.update(value="", interactive=False)

def add_file(history, file):
    text = transcribe(
        file
    )
    
    
    history = history + [(text, None)]
    return history


def bot(history):

   
    history[-1][1] = ""
    for character in text_client.predict(
                    history,
                    system_message,
                    temperature,
                    4096,
                    temperature,
                    repetition_penalty,
                    api_name="/chat"
    ):
        history[-1][1] += character
        yield history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png"))),
    )

    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter text and press enter, or speak to your microphone",
            container=False,
        )
        btn = gr.inputs.Audio(source="microphone", type="filepath", optional=True)

    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, chatbot
    )
    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)
    
    file_msg = btn.stop_recording(add_file, [chatbot, btn], [chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    
    #file_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)

demo.queue()
demo.launch(debug=True)