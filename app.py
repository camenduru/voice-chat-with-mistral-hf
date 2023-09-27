from __future__ import annotations

import os
# By using XTTS you agree to CPML license https://coqui.ai/cpml
os.environ["COQUI_TOS_AGREED"] = "1"

import gradio as gr
import numpy as np
import torch
import nltk  # we'll use this to split into sentences
nltk.download('punkt')
import uuid
import soundfile as SF

from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1", gpu=True)

title = "Speak with Llama2 70B"

DESCRIPTION = """# Speak with Llama2 70B"""
css = """.toast-wrap { display: none !important } """


os.environ["GRADIO_TEMP_DIR"] = "/home/yoach/spaces/tmp"

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


def add_text(history, text, agree):
    history = [] if history is None else history
    history = history + [(text, None)]
    return history, gr.update(value="", interactive=False)


def add_file(history, file, agree):
    history = [] if history is None else history
    text = transcribe(
        file
    )
    
    history = history + [(text, None)]
    return history



def bot(history, agree, system_prompt=""):    
    history = [] if history is None else history

    if system_prompt == "":
        system_prompt = system_message
        
    history[-1][1] = ""
    for character in text_client.submit(
                    history,
                    system_prompt,
                    temperature,
                    4096,
                    temperature,
                    repetition_penalty,
                    api_name="/chat"
    ):
        history[-1][1] = character
        yield history  

    
def generate_speech(history):
    text_to_generate = history[-1][1]
    text_to_generate = text_to_generate.replace("\n", " ").strip()
    text_to_generate = nltk.sent_tokenize(text_to_generate)
    
    filename = f"{uuid.uuid4()}.wav"
    sampling_rate = tts.synthesizer.tts_config.audio["sample_rate"]
    silence = [0] * int(0.25 * sampling_rate)

    
    for sentence in text_to_generate:
        # generate speech by cloning a voice using default settings
        wav = tts.tts(text=sentence,
                    speaker_wav="examples/female.wav",
                    decoder_iterations=20,
                    speed=1.2,
                    language="en")
        
        yield (sampling_rate, np.array(wav)) #np.array(wav + silence))

        

with gr.Blocks(title=title) as demo:
    gr.Markdown(DESCRIPTION)
    
    
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        avatar_images=('examples/lama.jpeg', 'examples/lama2.jpeg'),
        bubble_full_width=False,
    )

    with gr.Row():
        txt = gr.Textbox(
            scale=1,
            show_label=False,
            placeholder="Enter text and press enter, or speak to your microphone",
            container=False,
        )
        btn = gr.Audio(source="microphone", type="filepath", scale=2)
        
    with gr.Row():
        audio = gr.Audio(type="numpy", streaming=True, autoplay=True, label="Generated audio response", show_label=True)

    clear_btn = gr.ClearButton([chatbot, audio])
    

    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, chatbot
    ).then(generate_speech, chatbot, audio)
    
    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)
    
    file_msg = btn.stop_recording(add_file, [chatbot, btn], [chatbot], queue=False).then(
        bot, chatbot, chatbot
    ).then(generate_speech, chatbot, audio)
    

    gr.Markdown("""
This Space demonstrates how to speak to a chatbot, based solely on open-source models.
It relies on 3 models:
1. [Whisper-large-v2](https://huggingface.co/spaces/sanchit-gandhi/whisper-large-v2) as an ASR model, to transcribe recorded audio to text. It is called through a [gradio client](https://www.gradio.app/docs/client).
2. [Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) as the chat model, the actual chat model. It is also called through a [gradio client](https://www.gradio.app/docs/client).
3. [Coqui's XTTS](https://huggingface.co/spaces/coqui/xtts) as a TTS model, to generate the chatbot answers. This time, the model is hosted locally.

Note:
- As a derivate work of [Llama-2-70b-chat](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) by Meta,
this demo is governed by the original [license](https://huggingface.co/spaces/ysharma/Explore_llamav2_with_TGI/blob/main/LICENSE.txt) and [acceptable use policy](https://huggingface.co/spaces/ysharma/Explore_llamav2_with_TGI/blob/main/USE_POLICY.md).
- By using this demo you agree to the terms of the Coqui Public Model License at https://coqui.ai/cpml""")
demo.queue()
demo.launch(debug=True)