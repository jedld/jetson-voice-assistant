
import numpy as np
import time
from openwakeword.model import Model

owwModel = Model()

from pvrecorder import PvRecorder
audio = []
device_index = 0

for index, device in enumerate(PvRecorder.get_available_devices()):
    print(f"[{index}] {device}")
    if "Blue Snowball" in device:
        device_index = index
recorder = PvRecorder(device_index=device_index, frame_length=1280)

silence_threshold = 0.01  # This value might need adjustment
silence_duration = 3.0  # Duration of silence in seconds
silence_timer = None

try:
    recorder.start()
    print("recording...")
    while True:
        frame = recorder.read()
        np_frame = np.array(frame, dtype=np.int16)
        np_frame = np_frame.astype(np.float32) / 32768.0
        audio.append(np_frame)

        # Check if current frame is silence
        if np.all(np.abs(np_frame) < silence_threshold):
            print("_", end="", flush=True)
            if silence_timer is None:
                silence_timer = time.time()
            elif time.time() - silence_timer > silence_duration:
                break  # Break the loop after 3 seconds of silence
        else:
            print("^", end="", flush=True)
            silence_timer = None  # Reset timer if noise is detected
except Exception as e:
    print(e)
finally:
    recorder.stop()
    print("stopped recording")
    audio_data = np.concatenate(audio)
    prediction = owwModel.predict(audio)
    print(prediction)
    recorder.delete()    
