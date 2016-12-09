# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np
import pandas as pd
import librosa
import glob
import os
import string
import itertools
import pdb

__author__ = 'buriburisuri@gmail.com'


class ComTrans(object):

    def __init__(self, batch_size=32, name='train'):

        # load train corpus
        sources, targets = self._load_corpus(mode='train')

        # to constant tensor
        source = tf.convert_to_tensor(sources)
        target = tf.convert_to_tensor(targets)

        # create queue from constant tensor
        source, target = tf.train.slice_input_producer([source, target])

        # create batch queue
        batch_queue = tf.train.shuffle_batch([source, target], batch_size,
                                             num_threads=32, capacity=batch_size*64,
                                             min_after_dequeue=batch_size*32, name=name)

        # split data
        self.source, self.target = batch_queue

        # calc total batch count
        self.num_batch = len(sources) // batch_size

        # print info
        tf.sg_info('Train data loaded.(total data=%d, total batch=%d)' % (len(sources), self.num_batch))

    def _load_corpus(self, mode='train'):

        # load en-fr parallel corpus
        from nltk.corpus import comtrans
        als = comtrans.aligned_sents('alignment-en-fr.txt')

        # make character-level parallel corpus
        all_byte, sources, targets = [], [], []
        for al in als:
            src = [ord(ch) for ch in ' '.join(al.words)]  # source language byte stream
            tgt = [ord(ch) for ch in ' '.join(al.mots)]  # target language byte stream
            sources.append(src)
            targets.append(tgt)
            all_byte.extend(src + tgt)

        # make vocabulary
        self.index2byte = [0, 1] + list(np.unique(all_byte))  # add <EMP>, <EOS> tokens
        self.byte2index = {}
        for i, b in enumerate(self.index2byte):
            self.byte2index[b] = i
        self.voca_size = len(self.index2byte)
        self.max_len = 150

        # remove short and long sentence
        src, tgt = [], []
        for s, t in zip(sources, targets):
            if 50 <= len(s) < self.max_len and 50 <= len(t) < self.max_len:
                src.append(s)
                tgt.append(t)

        # convert to index list and add <EOS> to end of sentence
        for i in range(len(src)):
            src[i] = [self.byte2index[ch] for ch in src[i]] + [1]
            tgt[i] = [self.byte2index[ch] for ch in tgt[i]] + [1]

        # zero-padding
        for i in range(len(tgt)):
            src[i] += [0] * (self.max_len - len(src[i]))
            tgt[i] += [0] * (self.max_len - len(tgt[i]))

        # swap source and target : french -> english
        return tgt, src

    def to_batch(self, sentences):

        # convert to index list and add <EOS> to end of sentence
        for i in range(len(sentences)):
            sentences[i] = [self.byte2index[ord(ch)] for ch in sentences[i]] + [1]

        # zero-padding
        for i in range(len(sentences)):
            sentences[i] += [0] * (self.max_len - len(sentences[i]))

        return sentences

    def print_index(self, indices):
        for i, index in enumerate(indices):
            str_ = ''
            for ch in index:
                if ch > 1:
                    str_ += unichr(self.index2byte[ch])
                elif ch == 1:  # <EOS>
                    break
            print '[%d]' % i + str_


class TedTrans(object):

    def __init__(self, batch_size=16, fs=44100, data_path='asset/ted/', name='train'):

        @tf.sg_producer_func
        def _load_mfcc(src_list):
            lab, wav = src_list  # label, wave_file
            # load wave file
            wav, sr = librosa.load(wav, mono=True)
            # mfcc
            mfcc = librosa.feature.mfcc(wav, sr)
            # return result
            return lab, mfcc

        self.fs = fs;     

        # load corpus
        targets, wave_files = self._load_corpus(data_path)

        # to constant tensor
        target = tf.convert_to_tensor(targets)
        wave_file = tf.convert_to_tensor(wave_files)

        # create queue from constant tensor
        target, wave_file = tf.train.slice_input_producer([target, wave_file], shuffle=True)

        # decode wave file
        target, mfcc = _load_mfcc(source=[target, wave_file], dtypes=[tf.sg_intx, tf.sg_floatx],
                                 capacity=128, num_threads=32)

        # create batch queue with dynamic pad
        batch_queue = tf.train.batch([target, mfcc], batch_size,
                                     shapes=[(self.max_len,), (20, None)],
                                     num_threads=32, capacity=batch_size*64,
                                     dynamic_pad=True)

        # split data
        self.target,  self.mfcc = batch_queue
        # batch * time * dim
        self.mfcc = self.mfcc.sg_transpose(perm=[0, 2, 1])

        # calc total batch count
        self.num_batch = len(targets) // batch_size

        # print info
        tf.sg_info('TED EN-CN corpus loaded.(total data=%d, total batch=%d)' % (len(targets), self.num_batch))

    def _load_corpus(self, data_path):

        # make file ID
        file_ids = []
        for d in [data_path + 'text.en/']:
            file_ids.extend([f[18:-4] for f in sorted(glob.glob(d + '*.txt'))])

        # make wave file list
        wav_files = [data_path + 'wav.en/' + f + '.wav' for f in file_ids]

        # exclude extremely short wave files
        file_id, wav_file = [], []
        self.min_duration = 0;
        self.max_duration = 10; # to be optimally determined based on our data, and pooling size later. CY
        for i, w in zip(file_ids, wav_files):
            if os.stat(w).st_size > self.fs * self.min_duration and os.stat(w).st_size < self.fs * self.max_duration: 
                file_id.append(i)
                wav_file.append(w)

        all_byte, targets = [], []
        for f in file_id:
            # remove punctuation, to lower
            s = ' '.join(open(data_path + 'text.en/' + f + '.txt').read()
                         .translate(None, string.punctuation).lower().split())
            tgt = [ord(ch) for ch in s]
            targets.append(tgt)
            all_byte.extend(tgt)

        self.index2byte = [0, 1] + list(np.unique(all_byte))  # add <EMP>, <EOS> tokens  
        self.byte2index = {}

        for i, b in enumerate(self.index2byte):
            self.byte2index[b] = i 
        self.voca_size = len(self.index2byte)
        self.min_len =  0
        self.max_len =  100

        tgt, wav = [], []
        for t, w in zip(targets, wav_files):
            if len(t) > self.min_len and len(t) < self.max_len:
                tgt.append(t)
                wav.append(w)

        # convert to index list and add <EOS> to end of sentence
        for i in range(len(tgt)):
            tgt[i] = [self.byte2index[ch] for ch in tgt[i]] + [1] 

        # zero-padding
        for i in range(len(tgt)):
            tgt[i] += [0] * (self.max_len - len(tgt[i]))

        return tgt, wav
        
    def print_index(self, indices):
        # transform label index to character
        for i, index in enumerate(indices):
            str_ = ''
            for ch in index:
                if ch > 0:
                    str_ += unichr(self.index2byte[ch])
                elif ch == 0:  # <EOS>
                    break
            print str_