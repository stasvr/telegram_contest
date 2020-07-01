import os
import json
import codecs
from makers import *
from bs4 import BeautifulSoup
from utils import TextPreprocessor
from gensim.corpora import Dictionary

class Estimator(object):
    
    def __init__(self, source):
        self.data, self.names, self.titles = self._load_data(source)
        self.text_processor = TextPreprocessor()
        
    def _load_data(self, source):
        items, titles = list(), list()
        names = os.listdir(source)
        for file in names:
            if file.endswith(".html"):
                f = codecs.open(source + file, 'r', 'utf-8')
                bs = BeautifulSoup(f.read())
                
                document = bs.get_text()
                items.append(document)

                if not bs.title:
                    title = bs.find("meta",  property="og:title")["content"]
                else:
                    title = bs.title
                titles.append(title)
                
        return items, names, titles
    
class EstimatorCategory(Estimator): # uncomment
    
    def __init__(self, source):
        super(EstimatorCategory, self).__init__(source)
        self.word2index = Dictionary.load('../thirdparty/lda_model.id2word')
        with open('../thirdparty/category2index.json', 'r', encoding='utf-8') as f:
            self.theme2index = json.load(f)
        self.inverse_theme2index = {j: i for i, j in zip(self.theme2index.keys(), self.theme2index.values())}
        self.model_category = CategoryMaker(path_h5='../thirdparty/weights.h5')
            
    def run(self):
        tmp_data = self.__prepare_text()
        tmp_data = self.__prepare_category(tmp_data)
        return self.__prepare_out(tmp_data)
        
    def __prepare_text(self):
        tmp_data = [self.text_processor.clear_text(text) for text in self.data]
        tmp_data = [self.text_processor.tokenizer(text) for text in tmp_data]
        return tmp_data
        
    def __prepare_category(self, data):
        
        self.feed_matrix = np.zeros(shape=(len(data), 100))
        
        def vectorize(sentence, n):
            stem = self.text_processor.morph['en'].stem
            for idx, w in enumerate(sentence[:100]):
                if stem(w) in self.word2index:
                    self.feed_matrix[n, idx] = self.word2index[w]
                else:
                    self.feed_matrix[n, idx] = 0
        
        for idx, sentence in enumerate(data):
            vectorize(sentence, idx)
            
        tmp_data = self.model_category.predict(self.feed_matrix)
        tmp_data = np.argmax(tmp_data, axis=0)
        return tmp_data
    
    def __prepare_out(self, data): # make output
        out = [{'category': category, 'articles': list()} for category in self.theme2index]
        
        for category_idx, file in zip(data, self.names):
            for buff in out:
                if self.inverse_theme2index[int(category_idx)] == buff['category']:
                    buff['articles'].append(file)
        
        return json.dumps(out, indent=4, sort_keys=True)
    
class EstimatorTopics(Estimator):
    
    def __init__(self, source):
        super(EstimatorTopics, self).__init__(source)
        self.maxlen = 100
        self.model_topic = TopicMaker(path_lda='../thirdparty/')
            
    def run(self):
        tmp_data = self.__prepare_text()
        tmp_data = self.__prepare_topics(tmp_data)
        return self.__prepare_out(tmp_data)
        
    def __prepare_text(self):
        tmp_data = [self.text_processor.clear_text(text) for text in self.data]
        tmp_data = [self.text_processor.tokenizer(text) for text in tmp_data]
        tmp_data = [self.model_topic.doc2bow(text) for text in tmp_data]
        tmp_data = [self.model_topic.get_vector(el) for el in tmp_data]
        return tmp_data
        
    def __prepare_topics(self, data):
        return self.model_topic.get_cluster(data)
    
    def __prepare_out(self, data):
        out = [{'title': title, 'articles': list()} for title in data[0]]
        
        for i, file in zip(data[1], self.names):
            for buff in out:
                if i == buff['title']:
                    buff['articles'].append(file)
        
        return json.dumps(out, indent=4, sort_keys=True)
    
class EstimatorLanguage(Estimator):
    
    def __init__(self, source):
        super(EstimatorLanguage, self).__init__(source)
        self.threshold = 0.1
        
    def run(self):
        tmp_data = self.__prepare_text()
        tmp_data = self.__prepare_language(tmp_data)
        return self.__prepare_out(tmp_data)
        
    def __prepare_text(self):
        tmp_data = [self.text_processor.clear_text(text, replacer='') for text in self.data]
        return tmp_data
        
    def __prepare_language(self, data):
        def predict(text, size):
            res = self.text_processor.get_language(text, size)

            if res['other'] >= self.threshold:
                return 'other'
            else:
                if res['ru'] > res['en']:
                    return 'ru'
                else:
                    return 'en'
                
        tmp_data = [predict(text, len(text)) for text in data]
        return tmp_data
    
    def __prepare_out(self, data):
        out = [
            {
                "lang_code": "ru",
                "articles": []
            },
            {
                "lang_code": "en",
                "articles": []
            }
        ]
        for lang, file in zip(data, self.names):
            if lang == 'run':
                out[0]["articles"].append(file)
            if lang == 'en':
                out[1]["articles"].append(file)
        return json.dumps(out, indent=4, sort_keys=True)