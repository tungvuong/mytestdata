from pathlib import Path
import pandas as pd
import numpy as np
import json
import os
from os.path import exists
import math
import random
import re
import argparse


"""
Text generation using a Recurrent Neural Network (LSTM).
"""

import argparse
import os
import random
import time

import numpy as np
import tensorflow as tf


tf.compat.v1.disable_eager_execution()

class ModelNetwork:
    """
    RNN with num_layers LSTM layers and a fully-connected output layer
    The network allows for a dynamic number of iterations, depending on the
    inputs it receives.

       out   (fc layer; out_size)
        ^
       lstm
        ^
       lstm  (lstm size)
        ^
        in   (in_size)
    """
    def __init__(self, in_size, lstm_size, num_layers, out_size, session,
                 learning_rate=0.003, name="rnn"):
        self.scope = name
        self.in_size = in_size
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.out_size = out_size
        self.session = session
        self.learning_rate = tf.constant(learning_rate)
        # Last state of LSTM, used when running the network in TEST mode
        self.lstm_last_state = np.zeros(
            (self.num_layers * 2 * self.lstm_size,)
        )
        with tf.compat.v1.variable_scope(self.scope):
            # (batch_size, timesteps, in_size)
            self.xinput = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None, None, self.in_size),
                name="xinput"
            )
            self.lstm_init_value = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None, self.num_layers * 2 * self.lstm_size),
                name="lstm_init_value"
            )
            # LSTM
            self.lstm_cells = [
                tf.compat.v1.nn.rnn_cell.BasicLSTMCell(
                    self.lstm_size,
                    forget_bias=1.0,
                    state_is_tuple=False
                ) for i in range(self.num_layers)
            ]
            self.lstm = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                self.lstm_cells,
                state_is_tuple=False
            )
            # Iteratively compute output of recurrent network
            outputs, self.lstm_new_state = tf.compat.v1.nn.dynamic_rnn(
                self.lstm,
                self.xinput,
                initial_state=self.lstm_init_value,
                dtype=tf.float32
            )
            # Linear activation (FC layer on top of the LSTM net)
            self.rnn_out_W = tf.Variable(
                tf.compat.v1.random_normal(
                    (self.lstm_size, self.out_size),
                    stddev=0.01
                )
            )
            self.rnn_out_B = tf.Variable(
                tf.compat.v1.random_normal(
                    (self.out_size,), stddev=0.01
                )
            )
            outputs_reshaped = tf.reshape(outputs, [-1, self.lstm_size])
            network_output = tf.matmul(
                outputs_reshaped,
                self.rnn_out_W
            ) + self.rnn_out_B
            batch_time_shape = tf.shape(outputs)
            self.final_outputs = tf.reshape(
                tf.nn.softmax(network_output),
                (batch_time_shape[0], batch_time_shape[1], self.out_size)
            )
            # Training: provide target outputs for supervised training.
            self.y_batch = tf.compat.v1.placeholder(
                tf.float32,
                (None, None, self.out_size)
            )
            y_batch_long = tf.reshape(self.y_batch, [-1, self.out_size])
            self.cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=network_output,
                    labels=y_batch_long
                )
            )
            self.train_op = tf.compat.v1.train.RMSPropOptimizer(
                self.learning_rate,
                0.9
            ).minimize(self.cost)

    # Input: X is a single element, not a list!
    def run_step(self, x, init_zero_state=True):
        # Reset the initial state of the network.
        if init_zero_state:
            init_value = np.zeros((self.num_layers * 2 * self.lstm_size,))
        else:
            init_value = self.lstm_last_state
        out, next_lstm_state = self.session.run(
            [self.final_outputs, self.lstm_new_state],
            feed_dict={
                self.xinput: [x],
                self.lstm_init_value: [init_value]
            }
        )
        self.lstm_last_state = next_lstm_state[0]
        return out[0][0]

    # xbatch must be (batch_size, timesteps, input_size)
    # ybatch must be (batch_size, timesteps, output_size)
    def train_batch(self, xbatch, ybatch):
        init_value = np.zeros(
            (xbatch.shape[0], self.num_layers * 2 * self.lstm_size)
        )
        cost, _ = self.session.run(
            [self.cost, self.train_op],
            feed_dict={
                self.xinput: xbatch,
                self.y_batch: ybatch,
                self.lstm_init_value: init_value
            }
        )
        return cost


def embed_to_vocab(data_, vocab):
    """
    Embed string to character-arrays -- it generates an array len(data)
    x len(vocab).

    Vocab is a list of elements.
    """
    data = np.zeros((len(data_), len(vocab)))
    cnt = 0
    for s in data_:
        v = [0.0] * len(vocab)
        v[vocab.index(s)] = 1.0
        data[cnt, :] = v
        cnt += 1
    return data


def decode_embed(array, vocab):
    return vocab[array.index(1)]


def load_data(input):
    # Load the data
    data_ = ""
    with open(input, 'r') as f:
        data_ += f.read()
    data_ = data_.lower()
    # Convert to 1-hot coding
    vocab = sorted(list(set(data_)))
    data = embed_to_vocab(data_, vocab)
    return data, vocab


def check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('saved/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

def main():
    suggestions = {}
    with open('./mytestdata/queryindex.json') as json_file:
        queryindex = json.load(json_file)
    for filename in os.listdir("./mytestdata/3screens"):
        if filename.endswith(".csv"):
            tf.compat.v1.reset_default_graph()
            user = filename.replace('.csv','')
            print(user)
            suggestions[user] = []
            ratio = 0.7
            if user in ['D43D7EC3E0C2']:
                ratio = 0.85
            print('--------------',user,ratio,'----------')
            allindex = queryindex[filename.replace('.csv','')]
            splitindex = allindex[int(len(allindex)*ratio)]
            pred_index = allindex[int(len(allindex)*ratio):]
            print(splitindex)
            print(pred_index)
            data = pd.read_csv('./mytestdata/3screens/'+filename)
            #train, validate, test = np.split(data.sample(frac=1), [int(.6*len(data)), int(.8*len(data))])
            train, test = np.split(data.sample(frac=1), [int(ratio*len(data))])
            print(len(train))
            print(len(test))
            #rows = []
            #for row in train['source']:
            #    rows.append(row.split('</s>')[0])
            #with open('./mytestdata/lstm_data/'+user+'.txt', 'w') as f:
            #    for line in rows:
            #        f.write(f"{line[:500]}\n")
            # mkdir model train dir
            #Path("./mytestdata/lstm_data/saved/user").mkdir(parents=True, exist_ok=True)
            # Load the data
            data, vocab = load_data('./mytestdata/lstm_data/'+user+'.txt')
            
            in_size = out_size = len(vocab)
            lstm_size = 256  # 128
            num_layers = 2
            batch_size = 64  # 128
            time_steps = 100  # 50

            NUM_TRAIN_BATCHES = 20000
            
            # Number of test characters of text to generate after training the network
            LEN_TEST_TEXT = 500

            
            # Initialize the network
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.compat.v1.InteractiveSession(config=config)
            net = ModelNetwork(
                in_size=in_size,
                lstm_size=lstm_size,
                num_layers=num_layers,
                out_size=out_size,
                session=sess,
                learning_rate=0.003,
                name="char_rnn_network"
            )
            sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
            
            print('LOADING!')
            # 2) GENERATE LEN_TEST_TEXT CHARACTERS USING THE TRAINED NETWORK
            saver.restore(sess, './mytestdata/lstm_data/saved/'+user+'/model.ckpt')
            
            pred_df = pd.read_csv('./mytestdata/3screens/'+filename)[splitindex:]
            for index, row in pred_df.iterrows():
                #print('++++',row['title'],index)
                if (index+2 not in pred_index):
                    continue
                print()
                print(index+2)
                print('target',row['target'][:500])
                print('source',row['source'][:500])
                print('query: ',row['title'])
                TEST_PREFIX = row['source'].split('</s>')[0][:500]
                for i in range(len(TEST_PREFIX)):
                    try:
                        out = net.run_step(embed_to_vocab(TEST_PREFIX[i], vocab), i == 0)
                    except:
                        print('exception!')
                pred_target = ''
                for i in range(LEN_TEST_TEXT):
                    element = np.random.choice(range(len(vocab)), p=out)
                    pred_target += vocab[element]
                    out = net.run_step(embed_to_vocab(vocab[element], vocab), False)
                suggestions[user].append([row['title'],row['target'],pred_target,row['source'],index+2])
                print(pred_target)
            with open('./mytestdata/lstm3screen.json', 'w') as outfile:
                json.dump(suggestions, outfile)
            
#            TEST_PREFIX = ''
#            TARGET = ''
#            for row in test['source']:
#                TEST_PREFIX = row.split('</s>')[0]
#                break
#            for row in test['target']:
#                TARGET = row
#                break
#            for i in range(len(TEST_PREFIX)):
#                out = net.run_step(embed_to_vocab(TEST_PREFIX[i], vocab), i == 0)
#    
#            print("TARGET:")
#            print(TARGET)
#            print("SOURCE:")
#            print(TEST_PREFIX)
#            gen_str = TEST_PREFIX
#            for i in range(LEN_TEST_TEXT):
#                # Sample character from the network according to the generated
#                # output probabilities.
#                element = np.random.choice(range(len(vocab)), p=out)
#                gen_str += vocab[element]
#                out = net.run_step(embed_to_vocab(vocab[element], vocab), False)
#            print('GEN:')
#            print(gen_str)
            
            sess.close()
            #break

if __name__ == '__main__':
    main()  # execute this only when run directly, not when imported!   