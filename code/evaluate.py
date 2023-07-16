""" Evaluate BLEU Score 

"""





import tensorflow as tf
import tensorflow_text
from pickle import load, dump
from time import time 
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import gc
import os, sys 
import logging 





""" Tokenizer """ 

filename = os.path.join(os.path.dirname(sys.path[0]), 'metadata', 'tokenizer_deu_eng')
reloaded_tokenizers = tf.saved_model.load(filename)

""" Data """ 
filename = os.path.join(os.path.dirname(sys.path[0]), 'data', 'deu-eng-test.pkl')
with open(filename, 'rb') as file:
    test_data = load(file)
logging.info(f"Test Data Length = {len(test_data)}")

""" Load the model """
filename = os.path.join(os.path.dirname(sys.path[0]), 'metadata', 'translator')
translator = tf.saved_model.load(filename)





""" Evaluate """

actual, predicted = list(), list()
BLEU1, BLEU2, BLEU3, BLEU4, length = 0, 0, 0, 0, 0
filename = os.path.join(os.path.dirname(sys.path[0]), 'data', 'eng_processed.txt')
f = open( filename, 'rt')
texts = f.read()
texts = texts.strip().split('\n')
f.close()


time0 = time()
for i, source in enumerate(test_data):
    raw_src = source[0].decode('utf-8')
    # raw_target = source[1].decode('utf-8')
    raw_target = texts[i]
    # if i == 100: break
    #
    translation, _, _ = translator(raw_src)
    translation = translation.numpy().decode('utf-8')
    if i < 3: 
        logging.info(f"src = {raw_src}")
        logging.info(f"target = {raw_target}")
        logging.info(f"predict = {translation}")
        logging.info("\n")
    #
    actual.append([raw_target.split()])
    predicted.append(translation.split())
    
    length += 1
    if length % 200 ==0:
        logging.info(f"Finish = {length}, Duration = {time()-time0}sec")

 
logging.info(f'Predict time = {time()-time0}')
logging.info('BLEU-1 %f' % corpus_bleu(actual, predicted, weights=(1.0, 0.0, 0.0, 0.0)))
logging.info('BLEU-2 %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0.0, 0.0)))    
logging.info('BLEU-3 %f' % corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0.0))) 
logging.info('BLEU-4 %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))  
logging.info(f'BLEU time = {time()-time0}')

