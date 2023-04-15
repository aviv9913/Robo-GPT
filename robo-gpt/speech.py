import os
import tempfile
from threading import Thread
import requests
from playsound import playsound

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import numpy as np
import torch
import soundfile as sf
import simpleaudio as sa



BASE_URL = "https://api.elevenlabs.io/v1/text-to-speech"
VOICE_ID = "ErXwobaYiN019PkySvjV"
API_URL = f"{BASE_URL}/{VOICE_ID}"
API_KEY = os.getenv("ELEVENLABS_API_KEY")


def say_async(text: str) -> None:
    Thread(target=say, args=[text]).start()


def say(text: str) -> None:
    headers = {"Content-Type": "application/json", "xi-api-key": API_KEY}
    data = {"text": text}
    response = requests.post(API_URL, headers=headers, json=data)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(mode="wb") as temp:
            temp.write(response.content)
            temp.flush()
            playsound(temp.name)
    else:
        print(f"Failed to speak: status code = {response.status_code}\n{response.content}")


class SpeechT5Speaker:
    def __init__(self):
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

        speaker_embedding_array = np.load(r"C:\repos\Robo-GPT\robo-gpt\cmu_us_slt_arctic-wav-arctic_a0508.npy")
        self.speaker_embedding = torch.from_numpy(speaker_embedding_array).unsqueeze(0)

    def speak(self, text:str):
        tempfile_name = "speech_tmp.wav"
        inputs = self.processor(text=text, return_tensors="pt")
        speech = self.model.generate_speech(inputs["input_ids"], self.speaker_embedding, vocoder=self.vocoder)
        sf.write(tempfile_name, speech.numpy(), samplerate=16000)
        playsound(tempfile_name)
        try:
            os.remove(tempfile_name)
        except:
            pass


if __name__ == "__main__":
    speaker = SpeechT5Speaker()
    speaker.speak("Hello, my cat name is 'Sherry'")
    speaker.speak("And i'm your personal assistant")
