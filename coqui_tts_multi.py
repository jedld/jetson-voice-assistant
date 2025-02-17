import vocalizer
from TTS.api import TTS

class CoquiTTSMulti(vocalizer.Vocalizer):
    def __init__(self):
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
        print(TTS().list_models())

    def vocalize(self, prompt, language="en"):
        wav = self.tts.tts(text=prompt, speaker_wav="voices/leah.wav", language=language)
        return [wav, self.sample_rate()]
    
    def sample_rate(self):
        return 22050

    def is_multi_language(self):
        return True
    
    # English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt), Polish (pl), Turkish (tr), Russian (ru), Dutch (nl), Czech (cs), Arabic (ar), Chinese (zh-cn), Japanese (ja), Hungarian (hu), Korean (ko) Hindi (hi)
    def supported_languages(self):
        return ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"]