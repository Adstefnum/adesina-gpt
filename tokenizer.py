import re

class SimpleTokenizer:
    def __init__(self, vocab):
        self.text_to_tokens = vocab
        self.tokens_to_text = {idx: token for token, idx in vocab.items()}

    def encode(self, text):
         tokens = re.split(r'(\s+)|([,.;:?_()!"\']|--)', text)
         tokens = [token.strip() for token in tokens if token is not None and token.strip()]
         return [self.text_to_tokens[token] for token in tokens]

    def decode(self, ids):
        text = ' '.join([self.tokens_to_text[idx] for idx in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
        
    