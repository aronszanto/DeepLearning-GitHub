from keras.layers.recurrent import LSTM
from keras.optimizers import Adagrad
from keras.losses import sparse_categorical_crossentropy
from keras.layers import Input, RepeatVector
from keras.models import Model
from keras.layers import Dense, Embedding
from keras.layers import LSTM, Activation



from sklearn.manifold import TSNE


"""
Implements an autoencoder that can be used to get sequential user embeddings
"""

class LSTMAuto():
    def __init__(self, inputlen, embsize, vocabsize, encodingsize):
        self.inputlen = inputlen
        self.embsize = embsize
        self.vocabsize = vocabsize
        self.encodingsize = encodingsize

        self.inputs = Input(shape=(inputlen,))
        self.embs = Embedding(output_dim=self.embsize,
                              input_dim=self.vocabsize,
                              input_length=self.inputlen)(self.inputs)

        self.encoded = LSTM(self.encodingsize)(self.embs)

        decoded = RepeatVector(self.inputlen)(self.encoded)
        decoded = LSTM(self.vocabsize, return_sequences=True)(decoded)

        self.activs = Activation(activation='softmax')(decoded)

        self.sequence_autoencoder = Model(self.inputs, self.activs)
        self.encoder = Model(self.inputs, self.encoded)
        self.sequence_autoencoder.compile(loss='categorical_crossentropy', optimizer='adagrad')

    def train(self, x,y, epochs=20, batch_size=20):
        self.sequence_autoencoder.fit(x,y,epochs=epochs, batch_size=batch_size)


    def encode(self,x):
        return self.encoder.predict(x)
