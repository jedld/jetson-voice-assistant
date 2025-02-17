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

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base.en")
vocalizer_class = os.environ.get("VOCALIZER_CLASS", "CoquiTTS")
VOCALIZER_CLASS =  CoquiTTS if vocalizer_class == "CoquiTTS" else CoquiTTSMulti if vocalizer_class == "CoquiTTSMulti" else BarkVocalizer

INPUT_DEVICE_NAME = os.environ.get("INPUT_DEVICE", default="Blue Snowball")
OUTPUT_DEVICE_NAME = os.environ.get("OUTPUT_DEVICE", default="USB PnP Audio Device")
OUTPUT_DEVICE_SAMPLE_RATE = int(os.environ.get("OUTPUT_DEVICE_SAMPLE_RATE", default=48000))
CONTEXT_LENGTH = int(os.environ.get("CONTEXT_LENGTH", default=340))

# If using the Yahboom Jetson Cube case with LED lights attached, set this to true
HAS_YAHBOOM_CASE = os.environ.get("HAS_YAHBOOM_CASE", default="false").lower() == "true"
MAX_FRAMES = 500
SYSTEM_MESSAGE = "You are MistralOrca, a large language model trained by Alignment Lab AI. Reply in a way as if you are in a voice conversation with a human. Make your answers short and concise."


# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Large Language Model
model = "/model/" + os.environ.get("LLM_MODEL", default="mistral-7b-openorca.Q2_K.gguf")
interaction_array = [
    {"role": "system", "content": SYSTEM_MESSAGE}
]

ctx = mp.get_context('spawn')
vocalization_queue = ctx.JoinableQueue(maxsize=50)
sound_queue = ctx.JoinableQueue()
inference_queue = ctx.JoinableQueue()
cancel_queue = ctx.JoinableQueue(maxsize=1)

def sound_worker(sound_queue: mp.Queue):
    devices = sd.query_devices()
    output_device_index = sd.default.device
    print(f"default device {output_device_index}")
    for device in devices:
        if OUTPUT_DEVICE_NAME in device['name']:
            output_device_index = device['index']
            break

    while True:
        try:
            sound, sample_rate = sound_queue.get()
            print(f"playing sound with sample rate {sample_rate}, output sample rate {OUTPUT_DEVICE_SAMPLE_RATE}")
            if sample_rate < OUTPUT_DEVICE_SAMPLE_RATE:
                # Calculate the number of samples in the upsampled data
                num_samples = round(len(sound) * OUTPUT_DEVICE_SAMPLE_RATE / sample_rate)

                # Resample the data
                sound = resample(sound, num_samples)
            sd.play(sound, device=output_device_index, samplerate=OUTPUT_DEVICE_SAMPLE_RATE, blocking=True)
            
        except Exception as e:
            print(e)
        finally:
            sound_queue.task_done()

def cancel_wakeword_worker(pyaudio_device_index, inference_queue: mp.Queue, cancel_queue: mp.Queue):
    audio_device = pyaudio.PyAudio()
    
    CHUNK = 2560
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    while True:
        request = cancel_queue.get()
        if request == "monitor":
            print("starting cancel wakeword monitoring .....")
            cancel_queue.task_done()
            oww_model = Model(enable_speex_noise_suppression=True)
            try:
                mic_stream = audio_device.open(format=FORMAT, channels=CHANNELS, input_device_index=pyaudio_device_index, rate=RATE, input=True, frames_per_buffer=CHUNK)
                while True:
                        try:
                            request = cancel_queue.get(block=False)
                            if request == "cancel":
                                print("cancelling cancel wakeword monitoring")
                                cancel_queue.task_done()
                                break
                        except Empty:
                            pass

                        audio = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)
                        # Feed to openWakeWord model
                        print("x", end="", flush=True)
                        oww_model.predict(audio)
                        if oww_model.prediction_buffer['alexa'][-1] > 0.5:
                            print("send cancel to inference")
                            inference_queue.put("cancel")
                            break

            finally:
                try:
                    mic_stream.stop_stream()
                    mic_stream.close()
                except Exception as e:
                    print(e)

def wait_for_vocalization():
    vocalization_queue.join()
    sound_queue.join()


# Should follow below format
# <|im_start|>system
# You are MistralOrca, a large language model trained by Alignment Lab AI. Write out your reasoning step-by-step to be sure you get the right answers!
# <|im_end|>
# <|im_start|>user
# How are you?<|im_end|>
# <|im_start|>assistant
# I am doing well!<|im_end|>
# <|im_start|>user
# Please tell me about how mistral winds have attracted super-orcas.<|im_end|>
# <|im_start|>assistant
def format_mistral_orca(messages):
    message_buffer = []
    for message in messages:
        if message["role"] == "user":
            message_buffer.append(f"<|im_start|>user\n{message['content']}<|im_end|>")
        elif message["role"] == "assistant":
            message_buffer.append(f"<|im_start|>assistant\n{message['content']}<|im_end|>")
        elif message["role"] == "system":
            message_buffer.append(f"<|im_start|>system\n{message['content']}<|im_end|>")
    message_buffer.append("<|im_start|>assistant\n")
    return "\n".join(message_buffer)

def interact(llm: Llama, messages):
    print(f"Messages: {messages}")
    global interaction_array
    
    llm_start = time.time()
    response_buffer = []

    try:
        # clear the inference queue
        # while not inference_queue.empty():
        #     inference_queue.get(block=False)

        # cancel_queue.put("monitor")
        prompt = format_mistral_orca(messages)

        # compute for tokenization length and truncate if necessary
        tokenized_prompt = llm.tokenizer().encode(prompt)

        # slide if prompt is greate than 3/4 of context length
        if len(tokenized_prompt) >= (CONTEXT_LENGTH * 3) // 4:
            tokenized_prompt = tokenized_prompt[-((CONTEXT_LENGTH * 3) // 4):]

        inference_start_time = time.time()
        for output in llm.create_completion(prompt=tokenized_prompt, stop="<|im_end|>", max_tokens=None, temperature=0.1, stream=True):
            try:
                if inference_start_time is not None:
                    print(f"inference in {time.time() - inference_start_time} seconds")
                    inference_start_time = None
                partial_text = output['choices'][0]["text"]
                vocalization_queue.put(partial_text)
                response_buffer.append(partial_text)
            except KeyError:
                pass
    except KeyboardInterrupt:
        print("inference cancelled")

    vocalization_queue.put("<s>")
    # cancel_queue.put("cancel")
    # cancel_queue.join()

    print(f"overall llm in {time.time() - llm_start} seconds")
    
    response_text = "".join(response_buffer)

    print(f"Final Response: {response_text}")
    if response_text == "":
        vocalization_queue.put("I'm sorry, I didn't quite catch that.")
    else:
        interaction_array.append({
            "role": "assistant",
            "content": response_text
        })
    print(f"waiting for vocalization to finish")
    
    wait_for_vocalization()
    
    print(f"vocalization finished")
    audio_as_np_int16, sample_rate = get_sound_as_np("chime.wav")
    sound_queue.put([audio_as_np_int16, sample_rate])

def vocalizaton_worker(vocalization_queue: mp.Queue, sound_queue: mp.Queue):
    tts = VOCALIZER_CLASS()
    
    def vocalize(text):
        global reference_speaker
        start_vocalize = time.time()
        # make sure text contains letters or numbers
        if not re.search('[a-zA-Z0-9]', text):
            return
        
        print(f"vocalizing: {text}")
        if tts.is_multi_language():
            language, prob = langid.classify(text)
            print("language: ", language, " prob: ", prob)
            if language in tts.supported_languages:
                wav, sr = tts.vocalize(text, language=language)
            else:
                wav, sr = tts.vocalize(text)
        else:
            wav, sr = tts.vocalize(text)
        
        print(f"vocalized in {time.time() - start_vocalize} seconds")
        sound_queue.put([wav, sr])

    vocalization_buffer = []
    MIN_PHRASE_LENGTH = 10
    def sanitize_string(str):
        str = str.replace("<|im_start|>assistant", "")
        pattern = r'[\[\]/\\(){}:|\x00-\x1F\x7F]|\n|\r'
        return re.sub(pattern, ' ', str).strip()
    
    while True:
        try:
            message = vocalization_queue.get()
            vocalization_buffer.append(message)
            # check for stop characters and immediately vocalize
            if ("<s>" in message):
                vocalization_str = "".join(vocalization_buffer)
                vocalization_str = vocalization_str.replace("<s>", "")
                response_text = sanitize_string(vocalization_str)
                
                # if response is not long enough pad it with spaces
                if len(response_text) < MIN_PHRASE_LENGTH:
                    response_text = response_text + " " * (MIN_PHRASE_LENGTH - len(response_text))

                vocalize(response_text)
                vocalization_buffer.clear()
            elif ("." in message or "?" in message or "," in message) and len(vocalization_buffer) > 0:
                vocalization_str = "".join(vocalization_buffer)

                def split_text_in_between(text, sep):
                    text_sections = text.split(sep)
                    text_sections = filter(lambda x: x.strip() != "", text_sections) 
                    return [f"{section}{sep} " for section in text_sections]

                for sep in ['.', '?', ',']:
                    if sep in vocalization_str:
                        v_str = split_text_in_between(vocalization_str, sep)
                        break

                for x in range(0, len(v_str)):
                    response_text = sanitize_string(v_str[x])
                    if (response_text == ""):
                        continue

                    # check if it isn't blank and is long enough
                    if response_text != "" and len(response_text) > MIN_PHRASE_LENGTH:
                        print(v_str[0])
                        response_text = response_text.replace(" AI ", " A.I. ")
                        vocalize(response_text)
                        vocalization_buffer.clear()
                        for index in range(x+1, len(v_str)):
                            vocalization_buffer.append(v_str[index])
        except Exception as e:
            print(e)
        finally:
            vocalization_queue.task_done()

start_conversation_threshold = 5.0 # Give the user 5 seconds to start a conversation before returning to standby
silence_threshold = 0.02  # This value might need adjustment
silence_duration = 2.0  # Duration of silence in seconds before determining if the user is done with his/her sentence
FOLLOW_UP_THRESHOLD = 5.0

def set_start_up_led(bot):
    if HAS_YAHBOOM_CASE:
        bot.set_RGB_Color(0)
        bot.set_RGB_Effect(1)

def set_listening_led(bot):
    if HAS_YAHBOOM_CASE:
        bot.set_RGB_Color(2)
        bot.set_RGB_Effect(1)

def set_processing_led(bot):
    if HAS_YAHBOOM_CASE:
        bot.set_RGB_Color(3)
        bot.set_RGB_Effect(1)

def set_standby_led(bot):
    if HAS_YAHBOOM_CASE:
        bot.set_RGB_Color(4)
        bot.set_RGB_Effect(1)

def set_voice_down_led(bot):
    if HAS_YAHBOOM_CASE:
        bot.set_RGB_Effect(0)
        bot.set_RGB_Color(2)

def set_voice_led(bot):
    if HAS_YAHBOOM_CASE:
        bot.set_RGB_Effect(1)

# Create a thread for image processing
def wait_for_wakeword(bot, audio_device, pyaudio_device_index):
    global silence_threshold

    print("waiting for wakeword")
    set_standby_led(bot)

    CHUNK = 2560
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    oww_model = Model(enable_speex_noise_suppression=True)
    mic_stream = audio_device.open(format=FORMAT, channels=CHANNELS, input_device_index=pyaudio_device_index, rate=RATE, input=True, frames_per_buffer=CHUNK)

    noise_sample = deque(maxlen=100)
    while True:
        audio = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)
        prediction = oww_model.predict(audio)
        if prediction['alexa'] < 0.8:
            break

    try:
        while True:
            # Get audio
            audio = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)
            # Feed to openWakeWord model
            print(".", end="", flush=True)
            prediction = oww_model.predict(audio)

            if prediction['alexa'] > 0.8:
                print(f"alexa {prediction['alexa']}")
                return noise_sample
            else:
                # Use frame to calibrate for ambient noise by 
                # setting threshold to 1.5x the average of the frame
                noise_sample.append(audio)
                # silence_threshold = (np.max(np.abs(audio)) * 1.1) / 32768.0
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        print("stopping mic stream")
        mic_stream.stop_stream()
        mic_stream.close()
        exit()
    finally:
        print("stopping mic stream")
        try:
            mic_stream.stop_stream()
            mic_stream.close()
        except Exception as e:
            print(e)
        # audio_device.terminate()

def start_speech_recognizer(whisper_model, llm, bot, audio_device, pyaudio_device_index, noise_data):
    global silence_threshold
    global interaction_array

    silence_timer = None
    follow_up_duration = None
    audio_buffer = deque(maxlen=MAX_FRAMES)

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 2560
    MAX_INACTIVITY = 6.0
    
    inactivity_timer = time.time()
    user_has_started_conversation = False
    while True:
        try:
            mic_stream = audio_device.open(format=FORMAT, channels=CHANNELS, input_device_index=pyaudio_device_index, rate=RATE, input=True, frames_per_buffer=CHUNK)
            print("recording...")
            set_listening_led(bot)

            silence_timer = time.time()
            while True:
                np_frame = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16).astype(np.float32)
                audio_buffer.append(np_frame)
                
                current_time = time.time()
                # Check if current frame is silence
                if np.all((np.abs(np_frame) / 32768.0 ) < 0.02):
                    print("_", end="", flush=True)
                    set_voice_down_led(bot)

                    if current_time - inactivity_timer > MAX_INACTIVITY:
                        try:
                            mic_stream.stop_stream()
                            mic_stream.close()
                        except Exception as e:
                            print(e)
                        return True
                    elif not user_has_started_conversation and current_time - silence_timer > start_conversation_threshold:
                        try:
                            mic_stream.stop_stream()
                            mic_stream.close()
                        except Exception as e:
                            print(e)
                        return True
                    elif time.time() - silence_timer > silence_duration:
                        break  # Break the loop after 3 seconds of silence
                else:
                    user_has_started_conversation = True
                    silence_timer = time.time()
                    inactivity_timer = time.time()
                    print("^", end="", flush=True)
                    set_voice_led(bot)
            try:
                mic_stream.stop_stream()
                mic_stream.close()
            except Exception as e:
                print(e)
        except KeyboardInterrupt:
            print("stopping mic stream")
            mic_stream.stop_stream()
            mic_stream.close()
            exit()
        except IOError as e: # Reset on Input Overflow errors
            try:
                mic_stream.stop_stream()
                mic_stream.close()
            except Exception as e:
                print(e)
            continue
        except Exception as e:
            print(e)


        print("stopped recording")
        
        audio_array = list(audio_buffer)
        audio_data = np.concatenate(audio_array)
        audio_buffer.clear()
        nr_start = time.time()
        # make sur noise_data has something
        if len(noise_data) == 0:
            return True
        
        noise_data_array = np.concatenate(list(noise_data))
        audio_data = nr.reduce_noise(y=audio_data, sr=RATE, y_noise=noise_data_array, use_torch=True)
        print(f"nr in {time.time() - nr_start} seconds")

        audio_data = audio_data.astype(np.float32) / 32768.0
        if np.all(np.abs(audio_data) < 0.02):
            if follow_up_duration is None:
                print("follow up threshold reached")
                return True
            elif (time.time() - follow_up_duration) > FOLLOW_UP_THRESHOLD:
                follow_up_duration = None
                return True
            
            print("no audio detected")
            silence_timer = None
            continue
        
        set_processing_led(bot)

        start_transcribe = time.time()
        result = whisper_model.transcribe(audio_data)
        print(f"transcribed in {time.time() - start_transcribe} seconds")

        # strip and normalize spaces
        text = result['text'].strip().replace("  ", " ")
        
        if text == "":
            continue

        if "end conversation" in text:
            return True
        
        # refresh follow duration
        if follow_up_duration != None:
            follow_up_duration = time.time()
        
        print(f"transcribed: {text} {follow_up_duration} {time.time()}")
       
        if follow_up_duration != None and (time.time() - follow_up_duration < FOLLOW_UP_THRESHOLD):
            print("interact with history ...")
            interaction_array.append({
                "role": "user",
                "content": text
            })
            interact(llm, interaction_array)
            follow_up_duration = time.time()
            inactivity_timer = time.time()
            silence_timer = None
        else:
            print("interact with no history ...")
            interaction_array = [
                {"role": "system", "content": SYSTEM_MESSAGE }
            ]
            interaction_array.append({
                "role": "user",
                "content": text
            })
            interact(llm, interaction_array)
            follow_up_duration = time.time()
            inactivity_timer = time.time()
            silence_timer = None

def get_sound_as_np(filename):
    ifile = wave.open(filename)
    samples = ifile.getnframes()
    audio = ifile.readframes(samples)                                                                               
    audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)
    sample_rate = ifile.getframerate()
    num_channels = ifile.getnchannels()

    # Check if the audio is stereo
    if num_channels == 2:
        # Reshape the array to separate channels and average them to mono
        audio_as_np_int16 = audio_as_np_int16.reshape(-1, 2).mean(axis=1).astype(np.int16)
    else:
        audio_as_np_int16 = audio_as_np_int16

    return audio_as_np_int16, sample_rate



if __name__ == '__main__':
    if HAS_YAHBOOM_CASE:
        bot = CubeNano(i2c_bus=7)
        set_start_up_led(bot)
    else:
        bot = None

    if VOCALIZER_CLASS == CoquiTTSMulti:
        CoquiTTSMulti() # just instantiate to download models and accept license

    # Instantiate the model(s)
    

    llm = Llama(model_path=model, n_ctx=CONTEXT_LENGTH, chat_format="llama-2", n_gpu_layers=9999)
    ws_m = whisper.load_model(WHISPER_MODEL)

    p = ctx.Process(target=vocalizaton_worker, args=(vocalization_queue, sound_queue))
    p.start()
    sound_thread = ctx.Process(target=sound_worker, args=(sound_queue,))  
    sound_thread.start()
    
    vocalization_queue.put("Hello! I am an advanced assistant and I am ready to Help!\n")
    vocalization_queue.put("You can call me \"Alexa\" to start a conversation.\n")
    wait_for_vocalization()


    audio_device = pyaudio.PyAudio()
    pyaudio_device_index = 0
    

    for i in range(audio_device.get_device_count()):
        dev = audio_device.get_device_info_by_index(i)
        if INPUT_DEVICE_NAME in dev['name']:
            pyaudio_device_index = i


    audio_as_np_int16, sample_rate = get_sound_as_np("chime.wav")

    while True:
        noise_data = wait_for_wakeword(bot, audio_device, pyaudio_device_index)
        sound_queue.put([audio_as_np_int16, sample_rate])
        start_speech_recognizer(ws_m, llm, bot, audio_device, pyaudio_device_index, noise_data)
