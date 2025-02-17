class Vocalizer:
    def vocalize(self, text):
        raise NotImplementedError("You must implement vocalize()")
    
    def sample_rate(self):
        raise NotImplementedError("You must implement sample_rate()")
    
    def is_multi_language(self):
        return False
    
    def supported_languages(self):
        return []