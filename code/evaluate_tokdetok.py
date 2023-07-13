
""" Toekize-Detokenize of test data for BLEU score 

"""





# Library
import tensorflow as tf
import tensorflow_text
from pickle import load, dump
from time import time 
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import gc
import os, sys





""" Load Tokenizer"""

model_name = os.path.join(os.path.dirname(sys.path[0]), 'metadata', 'tokenizer_deu_eng')
reloaded_tokenizers = tf.saved_model.load(model_name)

""" Load dataset """

filename = os.path.join(os.path.dirname(sys.path[0]), 'data', 'deu-eng-test.pkl')
with open(filename, 'rb') as file:
    test_data = load(file)
print(f"Test Data Length = {len(test_data)}")





""" Preprocess the test dataset 
Tokenize-and-detokenize the target sentence for later BLEU score

"""

filename = os.path.join(os.path.dirname(sys.path[0]), 'data', 'eng_processed.txt')
f = open( filename, 'a' )
length = 0
time0 = time()
for i, source in enumerate(test_data):
    # if i < 11980: continue
    # if i == 400: break
    raw_target = source[1].decode('utf-8')
    #
    raw_target = reloaded_tokenizers.eng.tokenize([raw_target])
    raw_target = reloaded_tokenizers.eng.detokenize(raw_target)
    raw_target = raw_target.numpy()[0].decode('utf-8')
    f.write(raw_target + '\n')
    #
    length += 1
    if length % 200 ==0:
        print(f"Finish = {length}, Duration = {time()-time0}sec")
        del reloaded_tokenizers
        gc.collect()
        model_name = os.path.join(os.path.dirname(sys.path[0]), 'metadata', 'tokenizer_deu_eng')
        reloaded_tokenizers = tf.saved_model.load(model_name)
        f.close()
        filename = os.path.join(os.path.dirname(sys.path[0]), 'data', 'eng_processed.txt')
        f = open( filename, 'a' )
f.close()

