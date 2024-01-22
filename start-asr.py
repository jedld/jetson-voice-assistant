import whisper
import numpy as np
import time

model = whisper.load_model("base")
from pvrecorder import PvRecorder
audio = []
for index, device in enumerate(PvRecorder.get_available_devices()):
    print(f"[{index}] {device}")

recorder = PvRecorder(device_index=4, frame_length=512)

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
    result = model.transcribe(audio_data)
    print(result)
    recorder.delete()    
