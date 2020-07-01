import re
import os
import json
from gensim.utils import tokenize
from gensim.parsing.porter import PorterStemmer
from pymorphy2 import MorphAnalyzer

class TextPreprocessor: # maybe get out language as siperate function
    
    def __init__(self):
        self.morph = {
            'ru' : MorphAnalyzer(),
            'en' : PorterStemmer()
        }
        self.other_significance = 10
        self.stopwords = dict()
        with open('../thirdparty/stop_ru.json', 'r', encoding='utf-8') as f:
            self.stopwords['ru'] = json.load(f)
        with open('../thirdparty/stop_en.json', 'r', encoding='utf-8') as f:
            self.stopwords['en'] = json.load(f)
        
    
    def get_language(self, text, size):
        en_len = len(re.findall("[a-zA-Z]", text))
        ru_len = len(re.findall("[а-яА-Я]", text))    
        other_len = size - en_len - ru_len

        if ru_len / size < 0.3 and en_len / size < 0.3:
            return {
            "en": 0.0,
            "ru": 0.0,
            "other": 1.0
        }

        ru_confidence = ru_len / size
        en_confidence = en_len / size
        other_confidence = other_len * self.other_significance / size

        return {
            "en": round(en_confidence, 3),
            "ru": round(ru_confidence, 3),
            "other": round(other_confidence , 3)
        }

    def clear_text(self, text, replacer=' '):
        text = re.sub("[ :\n\t.,!?_;*^=<>$()#~|+/@0-9]", replacer, text)
        text = re.sub(r'[^\w]', replacer, text)
        text = re.sub(' +', ' ', text)
        return text    
    
    def tokenizer(self, text):
        tokens = list(tokenize(text))
        tokens = [token for token in tokens if len(token) > 4]
        tokens = [token for token in tokens if token not in self.stopwords['en']]
        tokens = [self.morph['en'].stem(token) for token in tokens]
        return tokens