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

from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1", gpu=True)

title = "Voice chat with Mistral 7B Instruct"

DESCRIPTION = """# Voice chat with Mistral 7B Instruct"""
css = """.toast-wrap { display: none !important } """

repo_id = "ylacombe/voice-chat-with-lama"

system_message = "\nYou are a helpful, respectful and honest assistant. Your answers are short, ideally a few words long, if it is possible. Always answer as helpfully as possible, while being safe.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
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
from huggingface_hub import InferenceClient


whisper_client = Client("https://sanchit-gandhi-whisper-large-v2.hf.space/")
text_client = InferenceClient(
    "mistralai/Mistral-7B-Instruct-v0.1"
)


def format_prompt(message, history):
  prompt = "<s>"
  for user_prompt, bot_response in history:
    prompt += f"[INST] {user_prompt} [/INST]"
    prompt += f" {bot_response}</s> "
  prompt += f"[INST] {message} [/INST]"
  return prompt

def generate(
    prompt, history, temperature=0.9, max_new_tokens=256, top_p=0.95, repetition_penalty=1.0,
):
    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)

    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        seed=42,
    )

    formatted_prompt = format_prompt(prompt, history)

    stream = text_client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""

    for response in stream:
        output += response.token.text
        yield output
    return output


def transcribe(wav_path):
    
    return whisper_client.predict(
				wav_path,	# str (filepath or URL to file) in 'inputs' Audio component
				"transcribe",	# str in 'Task' Radio component
				api_name="/predict"
)
    

# Chatbot demo with multimodal input (text, markdown, LaTeX, code blocks, image, audio, & video). Plus shows support for streaming text.


def add_text(history, text):
    history = [] if history is None else history
    history = history + [(text, None)]
    return history, gr.update(value="", interactive=False)


def add_file(history, file):
    history = [] if history is None else history
    text = transcribe(
        file
    )
    
    history = history + [(text, None)]
    return history



def bot(history, system_prompt=""):    
    history = [] if history is None else history

    if system_prompt == "":
        system_prompt = system_message
        
    history[-1][1] = ""
    for character in generate(history[-1][0], history[:-1]):
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
        try:   

            # generate speech by cloning a voice using default settings
            wav = tts.tts(text=sentence,
                        speaker_wav="examples/female.wav",
                        decoder_iterations=25,
                        decoder_sampler="dpm++2m",
                        speed=1.2,
                        language="en")
            
            yield (sampling_rate, np.array(wav)) #np.array(wav + silence))

        except RuntimeError as e :
            if "device-side assert" in str(e):
                # cannot do anything on cuda device side error, need tor estart
                print(f"Exit due to: Unrecoverable exception caused by prompt:{sentence}", flush=True)
                gr.Warning("Unhandled Exception encounter, please retry in a minute")
                print("Cuda device-assert Runtime encountered need restart")
            else:
                print("RuntimeError: non device-side assert error:", str(e))
                raise e

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
            scale=3,
            show_label=False,
            placeholder="Enter text and press enter, or speak to your microphone",
            container=False,
        )
        txt_btn = gr.Button(value="Submit text",scale=1)
        btn = gr.Audio(source="microphone", type="filepath", scale=4)
        
    with gr.Row():
        audio = gr.Audio(type="numpy", streaming=True, autoplay=True, label="Generated audio response", show_label=True)

    clear_btn = gr.ClearButton([chatbot, audio])
    
    txt_msg = txt_btn.click(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, chatbot
    ).then(generate_speech, chatbot, audio)

    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)

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
2. [Mistral-7b-instruct](https://huggingface.co/spaces/osanseviero/mistral-super-fast) as the chat model, the actual chat model. It is called from [huggingface_hub](https://huggingface.co/docs/huggingface_hub/guides/inference).
3. [Coqui's XTTS](https://huggingface.co/spaces/coqui/xtts) as a TTS model, to generate the chatbot answers. This time, the model is hosted locally.

Note:
- By using this demo you agree to the terms of the Coqui Public Model License at https://coqui.ai/cpml""")
demo.queue()
demo.launch(debug=True, share=True)