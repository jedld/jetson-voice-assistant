import vocalizer
from TTS.api import TTS

class CoquiTTS(vocalizer.Vocalizer):
    def __init__(self):
        # self.tts_model = "tts_models/en/ljspeech/speedy-speech"
        self.tts_model = "tts_models/en/ljspeech/tacotron2-DDC_ph"
        self.tts = TTS(self.tts_model, gpu=True)
        print(TTS().list_models())

    def vocalize(self, prompt):
        return [self.tts.tts(text=f"{prompt}", split_sentences=False), self.sample_rate()]
    
    def sample_rate(self):
        return 22050
