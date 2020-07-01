import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from gensim.models.ldamodel import LdaModel
from sklearn.metrics import pairwise_distances_argmin_min

class TopicMaker():
    
    def __init__(self, path_lda):
        self.n_cluster = 2
        self._model_lda = LdaModel.load(path_lda + 'lda_model')
        self._model_knn = KMeans(n_clusters=self.n_cluster, random_state=42)
    
    def doc2bow(self, element):
        return self._model_lda.id2word.doc2bow(element)
        
    def get_vector(self, element):
        vector = np.zeros(shape=(self._model_lda.num_topics))
        for k in self._model_lda[element]: vector[k[0]] = k[1]
        return vector
    
    def get_cluster(self, data):
        self._model_knn.fit(data)
        
        clusters = self._model_knn.labels_
        centers = np.array(self._model_knn.cluster_centers_)
        
        idx_l = list()
        
        for cluster in range(self.n_cluster):
            cluster_vector = [centers[cluster]]
            tmp_vectors = np.array([i for i, j in zip(data, clusters) if j == cluster])
            idx = pairwise_distances_argmin_min(cluster_vector, tmp_vectors)[0][0]
            idx_l.append(float(idx))
            
        return idx_l, clusters.tolist()
       
class CategoryMaker():
    
    def __init__(self, path_h5): 
        self.maxlen = 100
        self.max_features = 365753
        self.model = self._construct()
        self.model.load_weights(path_h5)
        
    def _construct(self):
        input_tensor = tf.keras.layers.Input(shape=(self.maxlen, ))
        embeddings = tf.keras.layers.Embedding(
            self.max_features, 
            128, input_length = self.maxlen
        )(input_tensor)
        hidden_tensor = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(embeddings)
        hidden_tensor = tf.keras.layers.Dropout(0.5)(hidden_tensor)
        output_tensor = tf.keras.layers.Dense(7, activation='softmax')(hidden_tensor)

        model = tf.keras.Model(inputs = input_tensor, outputs = output_tensor)
        return model
    
    def predict(self, input_vector):
        assert input_vector.shape[1] == 100
        return self.model.predict(input_vector)