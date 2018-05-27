#!/usr/bin/env python3
# coding: utf-8
# File: lstm_train.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-5-22

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

class Translator:
    def __init__(self):
        self.batch_size = 64  # Batch size for training.
        self.epochs = 10  # Number of epochs to train for.
        self.latent_dim = 256  # Latent dimensionality of the encoding space.
        self.num_samples = 10000  # Number of samples to train on.
        self.data_path = 'data/translation_fra_en.txt'
        self.inputpath = 'model/source_words'
        self.outputpath = 'model/target_words'
        self.modelpath = 'model/seq2seq.h5'
        self.num_encoder_tokens, self.num_decoder_tokens, self.encoder_input_data, \
        self.decoder_input_data, self.decoder_target_data = self.build_data()

    '''构造训练数据'''
    def build_data(self):
        input_texts = []
        target_texts = []
        input_characters = set()
        target_characters = set()

        with open(self.data_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        #只采用10000条数据进行实验
        for line in lines[: min(self.num_samples, len(lines) - 1)]:
            input_text, target_text = line.split('\t')
            target_text = '\t' + target_text + '\n'
            input_texts.append(input_text)
            target_texts.append(target_text)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)

            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)

        input_characters = sorted(list(input_characters))
        target_characters = sorted(list(target_characters))

        with open(self.inputpath, 'w') as f:
            f.write("*".join([item for item in input_characters]))
        f.close()

        with open(self.outputpath, 'w') as f:
            f.write("*".join([item for item in target_characters]))
        f.close()

        num_encoder_tokens = len(input_characters)
        num_decoder_tokens = len(target_characters)

        max_encoder_seq_length = max([len(txt) for txt in input_texts])
        max_decoder_seq_length = max([len(txt) for txt in target_texts])

        input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
        target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

        encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
        decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
        decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, input_token_index[char]] = 1.

            for t, char in enumerate(target_text):
                decoder_input_data[i, t, target_token_index[char]] = 1.
                if t > 0:
                    decoder_target_data[i, t - 1, target_token_index[char]] = 1.

        return num_encoder_tokens, num_decoder_tokens, encoder_input_data, decoder_input_data, decoder_target_data

    '''构建模型'''
    def build_model(self):
        # 输入层
        encoder_inputs = Input(shape=(None, self.num_encoder_tokens))
        # LSTM编码层
        encoder = LSTM(self.latent_dim, return_state=True)
        # 特征抽象层
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # 保留encoder后得到的状态
        encoder_states = [state_h, state_c]
        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self.num_decoder_tokens))
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
        return model

    '''训练模型'''
    def train_model(self):
        model = self.build_model()
        model.fit([self.encoder_input_data, self.decoder_input_data], self.decoder_target_data,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  validation_split=0.2,
                  )
        model.save(self.modelpath)

tranlator = Translator()
tranlator.train_model()