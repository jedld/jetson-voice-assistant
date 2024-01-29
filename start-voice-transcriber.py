from llama_cpp import Llama
import torch
from collections import deque
import sounddevice as sd
import os
import whisper
import numpy as np
import time
import re
import multiprocessing as mp
from time import sleep
from openwakeword.model import Model
import pyaudio
from CubeNanoLib import CubeNano
from coqui_tts import CoquiTTS
from coqui_tts_multi import CoquiTTSMulti
from bark_vocalizer import BarkVocalizer
from queue import Empty
import noisereduce as nr
import langid
import librosa
import wave
from scipy.io import wavfile
from scipy.signal import resample
from googletrans import Translator, LANGUAGES

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")

INPUT_DEVICE_NAME = os.environ.get("INPUT_DEVICE", default="Blue Snowball")
OUTPUT_FILENAME = os.environ.get("OUTPUT_FILENAME", default="transcription.txt")

MAX_FRAMES = 500

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

ctx = mp.get_context('spawn')
sound_queue = ctx.JoinableQueue()
start_conversation_threshold = 5.0 # Give the user 5 seconds to start a conversation before returning to standby
silence_threshold = 0.02  # This value might need adjustment
silence_duration = 1.0  # Duration of silence in seconds before determining if the user is done with his/her sentence
FOLLOW_UP_THRESHOLD = 5.0

def transcribe_worker(sound_queue: mp.Queue):
    RATE = 16000
    whisper_model = whisper.load_model(WHISPER_MODEL)
    translator = Translator()
    with open(OUTPUT_FILENAME, "w") as f:
        while True:
            try:
                audio = sound_queue.get()
                audio_data = nr.reduce_noise(y=audio, sr=RATE, use_torch=True)
                audio_data = audio_data.astype(np.float32) / 32768.0
                if np.all(np.abs(audio_data) < 0.02):
                    continue
                
                
                result = whisper_model.transcribe(audio_data)

                # strip and normalize spaces
                text = result['text'].strip().replace("  ", " ")
                translated = translator.translate(text, src='ja', dest='en')
                f.write(translated.text + "\n")
                print("text:", translated.text)
            except Exception as e:
                print(e)
            finally:
                sound_queue.task_done()
                f.flush()


def start_speech_recognizer(sound_queue, audio_device, pyaudio_device_index):
    global silence_threshold
    global interaction_array

    silence_timer = None
    audio_buffer = deque(maxlen=MAX_FRAMES)

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 2560

    mic_stream = None
    while True:
        try:
            mic_stream = audio_device.open(format=FORMAT, channels=CHANNELS, input_device_index=pyaudio_device_index, rate=RATE, input=True, frames_per_buffer=CHUNK)
            silence_timer = time.time()
            while True:
                np_frame = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16).astype(np.float32)
                audio_buffer.append(np_frame)
                
                # Check if current frame is silence
                if np.all((np.abs(np_frame) / 32768.0 ) < 0.02):
                    if time.time() - silence_timer > silence_duration:
                        break  # Break the loop after 3 seconds of silence
                else:
                    silence_timer = time.time()
                   
            try:
                mic_stream.stop_stream()
                mic_stream.close()
            except Exception as e:
                print(e)
        except KeyboardInterrupt:
            mic_stream.stop_stream()
            mic_stream.close()
            exit()
        except Exception as e:
            print(e)

        
        audio_array = list(audio_buffer)
        audio_data = np.concatenate(audio_array)
        audio_buffer.clear()
        sound_queue.put(audio_data)
               

if __name__ == '__main__':

    p = ctx.Process(target=transcribe_worker, args=(sound_queue,))
    p.start()
        
    audio_device = pyaudio.PyAudio()
    pyaudio_device_index = 0
    
    for i in range(audio_device.get_device_count()):
        dev = audio_device.get_device_info_by_index(i)
        if INPUT_DEVICE_NAME in dev['name']:
            pyaudio_device_index = i

    while True:
        start_speech_recognizer(sound_queue, audio_device, pyaudio_device_index)
