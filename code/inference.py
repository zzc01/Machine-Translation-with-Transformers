""" Inference and Attention Plots

"""


import tensorflow as tf
import tensorflow_text
from pickle import load, dump
from time import time 
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import matplotlib.pyplot as plt
import sys, os
import logging 





""" Load dataset """
filename = os.path.join(os.path.dirname(sys.path[0]), 'data', 'deu-eng-test.pkl')
with open(filename, 'rb') as file:
    test_data = load(file)
type(test_data)    

""" Load Translator """
filename = os.path.join(os.path.dirname(sys.path[0]), 'metadata', 'translator_1')
translator = tf.saved_model.load(filename)

def print_translation(sentence, tokens, ground_truth):
    logging.info(f'{"Input:":15s}: {sentence}')
    logging.info(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
    logging.info(f'{"Ground truth":15s}: {ground_truth}')




""" Translate """

test_data2 = np.array(test_data)
trainX = test_data2[:, 0]
trainY = test_data2[:, 1]
logging.info(len(trainX))
for i in range(2000, 2005):
    if i == 10: break
    sentence = trainX[i].decode('utf-8')
    ground_truth = trainY[i].decode('utf-8')
    translated_text, translated_tokens, attention_weights = translator(tf.constant(sentence))
    print_translation(sentence, translated_text, ground_truth)





""" Attention Plots """


""" Load Tokenizer """
filename = os.path.join(os.path.dirname(sys.path[0]), 'metadata', 'tokenizer_deu_eng')
tokenizers = tf.saved_model.load(filename)

""" Plots Attention Heads """
def plot_attention_head(in_tokens, translated_tokens, attention):
  # The model didn't generate `<START>` in the output. Skip it.
  translated_tokens = translated_tokens[1:]

  ax = plt.gca()
  ax.matshow(attention)
  ax.set_xticks(range(len(in_tokens)))
  ax.set_yticks(range(len(translated_tokens)))

  labels = [label.decode('utf-8') for label in in_tokens.numpy()]
  ax.set_xticklabels(
      labels, rotation=90)

  labels = [label.decode('utf-8') for label in translated_tokens.numpy()]
  ax.set_yticklabels(labels)
#   plt.show()


sentence = "Fass nichts an, ohne zu fragen!"
ground_truth = "Don't touch anything without asking."
translated_text, translated_tokens, attention_weights = translator(tf.constant(sentence))
print_translation(sentence, translated_text, ground_truth)

head = 0
# Shape: `(batch=1, num_heads, seq_len_q, seq_len_k)`.
attention_heads = tf.squeeze(attention_weights, 0)
attention = attention_heads[head]
attention.shape

in_tokens = tf.convert_to_tensor([sentence])
in_tokens = tokenizers.deu.tokenize(in_tokens).to_tensor()
in_tokens = tokenizers.deu.lookup(in_tokens)[0]

logging.info(in_tokens)
logging.info(translated_tokens)
plot_attention_head(in_tokens, translated_tokens, attention)




""" Plots Attention Weights """
def plot_attention_weights(sentence, translated_tokens, attention_heads):
  in_tokens = tf.convert_to_tensor([sentence])
  in_tokens = tokenizers.deu.tokenize(in_tokens).to_tensor()
  in_tokens = tokenizers.deu.lookup(in_tokens)[0]

  fig = plt.figure(figsize=(16, 8))

  for h, head in enumerate(attention_heads):
    ax = fig.add_subplot(2, 4, h+1)
    plot_attention_head(in_tokens, translated_tokens, head)
    ax.set_xlabel(f'Head {h+1}')

  plt.tight_layout()
  plt.show()

plot_attention_weights(sentence,
                       translated_tokens,
                       attention_weights[0])





# sentence = 'Er war wütend auf seine Tochter.'
# ground_truth = 'He was angry with his daughter.'

# translated_text, translated_tokens, attention_weights = translator(tf.constant(sentence))
# print('\n')
# print_translation(sentence, translated_text, ground_truth)
# # Shape: `(batch=1, num_heads, seq_len_q, seq_len_k)`.
# plot_attention_weights(sentence, translated_tokens, attention_weights[0])

