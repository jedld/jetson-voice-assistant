import vocalizer
import os

from bark import SAMPLE_RATE, preload_models, generate_audio
class BarkVocalizer(vocalizer.Vocalizer):
    def __init__(self):
        os.environ["SUNO_OFFLOAD_CPU"] = "True"
        os.environ["SUNO_USE_SMALL_MODELS"] = "True"
        preload_models()

    def vocalize(self, prompt):
        return [generate_audio(prompt), self.sample_rate()]
    
    def sample_rate(self):
        return SAMPLE_RATE