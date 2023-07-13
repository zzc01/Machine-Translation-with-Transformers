
import tensorflow_text
import tensorflow as tf 
import os, sys 




def print_translation(sentence, tokens, ground_truth):
    print(f'{"Input:":15s}: {sentence}')
    print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
    print(f'{"Ground truth":15s}: {ground_truth}')





filename = os.path.join(os.path.dirname(sys.path[0]), 'metadata', 'translator')
reloaded = tf.saved_model.load(filename)

sentence = "Sie sind stolz darauf, Studenten jener Universität zu sein."
ground_truth = "They are proud to be students at that university."
translated_text, translated_tokens, attention_weights = reloaded(tf.constant(sentence))
print_translation(sentence, translated_text, ground_truth)
