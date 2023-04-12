'''
Part of PiTE
(c) 2023 by  Pengfei Zhang, Seojin Bang, Heewook Lee, and Arizona State University.
See LICENSE-CC-BY-NC-ND for licensing.
'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, GlobalMaxPooling1D, concatenate, Dense, BatchNormalization, Dropout
from tensorflow.keras.layers import Bidirectional, TimeDistributed, LSTM, Conv1D, Add
from tensorflow.math import subtract
from keras.models import Model


## Transformer Block
# https://keras.io/examples/nlp/text_classification_with_transformer/
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    
    def get_config(self):
        cfg = super().get_config()
        return cfg
    
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    

## The transformer-based model
embed_dim = 1024     # Embedding size for each token
num_heads = 2        # Number of attention heads
ff_dim = 32          # Hidden layer size in feed forward network inside transformer
def Transformer_based():
    X_tcr = Input(shape=(22,1024))
    X_epi = Input(shape=(22,1024))
    
    transformer_block_tcr = TransformerBlock(embed_dim, num_heads, ff_dim)
    transformer_block_epi = TransformerBlock(embed_dim, num_heads, ff_dim)
    
    sembed_tcr = transformer_block_tcr(X_tcr)
    sembed_tcr = tf.nn.silu(sembed_tcr)
    sembed_tcr = GlobalMaxPooling1D()(sembed_tcr)
    
    sembed_epi = transformer_block_epi(X_epi)
    sembed_epi = tf.nn.silu(sembed_epi)
    sembed_epi = GlobalMaxPooling1D()(sembed_epi)
    
    # concate [u, v, |u-v|]
    concate = concatenate([sembed_tcr, sembed_epi, abs(subtract(sembed_tcr,sembed_epi))])
    concate = Dense(1024)(concate)
    concate = BatchNormalization()(concate)
    concate = Dropout(0.3)(concate)
    concate = tf.nn.silu(concate)
    concate = Dense(1, activation='sigmoid')(concate)
    
    model = Model(inputs = [X_tcr, X_epi], outputs=concate, name='Transformer_based_model')
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam')
    print(model.summary())
    return model


## The BiLSTM-based model
def BiLSTM_based():
    X_tcr = Input(shape=(22,1024))
    X_epi = Input(shape=(22,1024))
    
    # sentence embedding encoder
    sembed_tcr = Bidirectional(LSTM(32, return_sequences=True))(X_tcr)
    sembed_tcr = TimeDistributed(Dense(256))(sembed_tcr)
    sembed_tcr = tf.nn.silu(sembed_tcr)
    sembed_tcr = GlobalMaxPooling1D()(sembed_tcr)
    
    sembed_epi = Bidirectional(LSTM(32, return_sequences=True))(X_epi)
    sembed_epi = TimeDistributed(Dense(256))(sembed_epi)
    sembed_epi = tf.nn.silu(sembed_epi)
    sembed_epi = GlobalMaxPooling1D()(sembed_epi)
    
    # concate [u, v, |u-v|]
    concate = concatenate([sembed_tcr, sembed_epi, abs(subtract(sembed_tcr,sembed_epi))])
    concate = Dense(1024)(concate)
    concate = BatchNormalization()(concate)
    concate = Dropout(0.3)(concate)
    concate = tf.nn.silu(concate)
    concate = Dense(1, activation='sigmoid')(concate)
    
    model = Model(inputs = [X_tcr, X_epi], outputs=concate, name='BiLSTM_based_model')
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam')
    print(model.summary())
    return model


## The ByteNet CNN based model
def byteNetEncoder():
    seq = Input(shape=(22,1024))
#     input_tensor = Flatten()(seq)
    encoder1 = Conv1D(filters=256, kernel_size=1, strides=1, padding="same")(seq);
    encoder1 = tf.nn.gelu(encoder1);
    encoder1 = Conv1D(filters=512, kernel_size=5, strides=1, padding="same", dilation_rate=1)(encoder1);
    encoder1 = BatchNormalization(axis=-1)(encoder1);
    encoder1 = tf.nn.gelu(encoder1);
    encoder1 = Conv1D(filters=1024, kernel_size=1, strides=1, padding="same")(encoder1);
    input_tensor = Add()([seq, encoder1]);
    encoder2 = BatchNormalization(axis=-1)(input_tensor);
    encoder2 = tf.nn.gelu(encoder2);
    encoder2 = Conv1D(filters=256, kernel_size=1, strides=1, padding="same")(input_tensor);
    encoder2 = BatchNormalization(axis=-1)(encoder2);
    encoder2 = tf.nn.gelu(encoder2);
    encoder2 = Conv1D(filters=512, kernel_size=5, strides=1, padding="same", dilation_rate=2)(encoder2);
    encoder2 = BatchNormalization(axis=-1)(encoder2);
    encoder2 = tf.nn.gelu(encoder2);
    encoder2 = Conv1D(filters=1024, kernel_size=1, strides=1, padding="same")(encoder2);
    input_tensor = Add()([input_tensor, encoder2])
    encoder3 = BatchNormalization(axis=-1)(input_tensor);
    encoder3 = tf.nn.gelu(encoder3);
    encoder3 = Conv1D(filters=256, kernel_size=1, strides=1, padding="same")(encoder3);
    encoder3 = BatchNormalization(axis=-1)(encoder3);
    encoder3 = tf.nn.gelu(encoder3);
    encoder3 = Conv1D(filters=512, kernel_size=5, strides=1, padding="same", dilation_rate=4)(encoder3);
    encoder3 = BatchNormalization(axis=-1)(encoder3);
    encoder3 = tf.nn.gelu(encoder3);
    encoder3 = Conv1D(filters=1024, kernel_size=1, strides=1, padding="same")(encoder3);
    input_tensor = Add()([input_tensor, encoder3])
    encoder4 = BatchNormalization(axis=-1)(input_tensor);
    encoder4 = tf.nn.gelu(encoder4);
    encoder4 = Conv1D(filters=256, kernel_size=1, strides=1, padding="same")(encoder4);
    encoder4 = BatchNormalization(axis=-1)(encoder4);
    encoder4 = tf.nn.gelu(encoder4);
    encoder4 = Conv1D(filters=512, kernel_size=5, strides=1, padding="same", dilation_rate=8)(encoder4);
    encoder4 = BatchNormalization(axis=-1)(encoder4);
    encoder4 = tf.nn.gelu(encoder4);
    encoder4 = Conv1D(filters=1024, kernel_size=1, strides=1, padding="same")(encoder4);
    input_tensor = Add()([input_tensor, encoder4])
    encoder5 = BatchNormalization(axis=-1)(input_tensor);
    encoder5 = tf.nn.gelu(encoder5);
    encoder5 = Conv1D(filters=256, kernel_size=1, strides=1, padding="same")(encoder5);
    encoder5 = BatchNormalization(axis=-1)(encoder5);
    encoder5 = tf.nn.gelu(encoder5);
    encoder5 = Conv1D(filters=512, kernel_size=5, strides=1, padding="same", dilation_rate=16)(encoder5);
    encoder5 = BatchNormalization(axis=-1)(encoder5);
    encoder5 = tf.nn.gelu(encoder5);
    encoder5 = Conv1D(filters=1024, kernel_size=1, strides=1, padding="same")(encoder5);
    input_tensor = Add()([input_tensor, encoder5])
    input_tensor = tf.nn.gelu(input_tensor);
    output_tensor = Conv1D(filters=1024, kernel_size=1, padding="same", activation="relu")(input_tensor)
    output_tensor = GlobalMaxPooling1D()(output_tensor)
    model = Model(inputs=seq, outputs=output_tensor, name='byteNetEncoder')
    return model


def byteNet_based():
    X_tcr = Input(shape=(22,1024))
    X_epi = Input(shape=(22,1024))
    
    byteNetencoder = byteNetEncoder()
    
    sembed_tcr = byteNetencoder(inputs=X_tcr)
    sembed_epi = byteNetencoder(inputs=X_epi)
    
    # concate [u, v, |u-v|]
    concate = concatenate([sembed_tcr, sembed_epi, abs(subtract(sembed_tcr,sembed_epi))])
    concate = Dense(1024)(concate)
    concate = BatchNormalization()(concate)
    concate = Dropout(0.3)(concate)
    concate = tf.nn.silu(concate)
    concate = Dense(1, activation='sigmoid')(concate)
    
    model = Model(inputs = [X_tcr, X_epi], outputs=concate, name='byteNet_based_model')
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam')
    print(model.summary())
    return model

def baseline_net():
    X_tcr = Input(shape=(1024,))
    X_epi = Input(shape=(1024,))
    
    # concate [u, v, |u-v|]
    concate = concatenate([X_tcr, X_epi, abs(subtract(X_tcr, X_epi))])
    concate = Dense(1024)(concate)
    concate = BatchNormalization()(concate)
    concate = Dropout(0.3)(concate)
    concate = tf.nn.silu(concate)
    concate = Dense(1, activation='sigmoid')(concate)
    
    model = Model(inputs=[X_tcr, X_epi], outputs=concate, name='baseline_model')
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam')
    print(model.summary())
    return model
