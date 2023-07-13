import pathlib
import tensorflow as tf
import tensorflow_text as tftxt
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
from pickle import dump, load
import numpy as np
import re
import sys, os
import logging 
from datetime import datetime


""" Tokenization is the process of breaking up text, into "tokens". 
Depending on the tokenizer, these tokens can represent sentence-pieces, words, subwords, or characters. 
Here use the subword tokenizer. 
We optimize two text.BertTokenizer objects (one for English, one for Portuguese) 
Then export them in a TensorFlow saved_model format.


"""





""" Create Vocab List """

class CreateVocab():
  def __init__(self, reserved_tokens):
    self.reserved_tokens = reserved_tokens 

  def __call__(self):
    logging.info("CreateVocab started")

    """ Load Data Set """
    filename = os.path.join(os.path.dirname(sys.path[0]), 'data', 'deu-eng-train.pkl')
    with open(filename, 'rb') as file:
      dataset = load(file)

    trainX = np.array(dataset)[:, 0]
    trainY = np.array(dataset)[:, 1]
    train_deu = tf.data.Dataset.from_tensor_slices(trainX)
    train_eng = tf.data.Dataset.from_tensor_slices(trainY)

    """ Create Vocabulary """
    bert_tokenizer_params = dict(lower_case=True)
    bert_tokenizer_params

    bert_vocab_args = dict(
        vocab_size = 8000,
        reserved_tokens = self.reserved_tokens,
        bert_tokenizer_params = bert_tokenizer_params,
        learn_params = {}
    )

    deu_vocab = bert_vocab.bert_vocab_from_dataset(
        train_deu.batch(1000).prefetch(2),
        **bert_vocab_args
    )

    eng_vocab = bert_vocab.bert_vocab_from_dataset(
        train_eng.batch(1000).prefetch(2),
        **bert_vocab_args
    )

    """ Save Vocab"""
    def write_vocab_file(filepath, vocab):
      with open(filepath, 'w', encoding="utf-8") as f:
        for token in vocab:
          print(token, file=f)

    filename = os.path.join(os.path.dirname(sys.path[0]), 'metadata', 'deu_vocab.txt')
    write_vocab_file(filename, deu_vocab)

    filename = os.path.join(os.path.dirname(sys.path[0]), 'metadata', 'eng_vocab.txt')
    write_vocab_file(filename, eng_vocab)

    logging.info("CreateVocab Finished")





""" Create Tokenizer """
class CreateTokenizer():
  def __init__(self, reserved_tokens):
    self.reserved_tokens = reserved_tokens

  def __call__(self):
    logging.info("Create Tokenizer Started")
    tokenizers = tf.Module()
    filename = os.path.join(os.path.dirname(sys.path[0]), 'metadata', 'deu_vocab.txt')
    tokenizers.deu = CustomTokenizer(self.reserved_tokens, filename)
    filename = os.path.join(os.path.dirname(sys.path[0]), 'metadata', 'eng_vocab.txt')
    tokenizers.eng = CustomTokenizer(self.reserved_tokens, filename)

    # Save The Model
    model_name = os.path.join(os.path.dirname(sys.path[0]), 'metadata', 'tokenizer_deu_eng')
    # model_name = path + 'tokenizer_deu_eng_1'
    tf.saved_model.save(tokenizers, model_name)
    logging.info("Create Tokenizer Finished")





class CustomTokenizer(tf.Module):
  def __init__(self, reserved_tokens, vocab_path):
    self.tokenizer = tftxt.BertTokenizer(vocab_path, lower_case=True)
    self._reserved_tokens = reserved_tokens
    self.START = tf.argmax(tf.constant(self._reserved_tokens) == "[START]")
    self.END = tf.argmax(tf.constant(self._reserved_tokens) == "[END]")
    self._vocab_path = tf.saved_model.Asset(vocab_path)
    #
    vocab = pathlib.Path(vocab_path).read_text(encoding="utf-8").splitlines()
    self.vocab = tf.Variable(vocab)

    ## Create signatures for export
    self.tokenize.get_concrete_function(
        tf.TensorSpec(shape=[None], dtype=tf.string))
    self.detokenize.get_concrete_function(
        tf.TensorSpec(shape=[None, None], dtype=tf.int64))
    self.detokenize.get_concrete_function(
        tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))
    self.lookup.get_concrete_function(
        tf.TensorSpec(shape=[None, None], dtype=tf.int64))
    self.lookup.get_concrete_function(
        tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))
    ##
    self.get_vocab_size.get_concrete_function()
    self.get_vocab_path.get_concrete_function()
    self.get_reserved_tokens.get_concrete_function()

  ##
  @tf.function
  def tokenize(self, strings):
    enc = self.tokenizer.tokenize(strings)
    enc = enc.merge_dims(-2,-1)
    enc = self.add_start_end(enc)
    return enc

  @tf.function
  def detokenize(self, tokenized):
    words = self.tokenizer.detokenize(tokenized)
    words = self.cleanup_text(self._reserved_tokens, words)
    return words

  @tf.function
  def lookup(self, token_ids):
    return tf.gather(self.vocab, token_ids)

  @tf.function
  def get_vocab_size(self):
    return tf.shape(self.vocab)[0]

  @tf.function
  def get_vocab_path(self):
    return self._vocab_path

  @tf.function
  def get_reserved_tokens(self):
    return tf.constant(self._reserved_tokens)


  # ADD START, END
  def add_start_end(self, ragged):
    count = ragged.bounding_shape()[0]
    starts = tf.fill([count, 1], self.START)
    ends = tf.fill([count, 1], self.END)
    return tf.concat([starts, ragged, ends], axis=1)

  # Cleanup detokenized text
  def cleanup_text(self, reserved_tokens, token_txt):
    bad_tokens = [re.escape(tok) for tok in reserved_tokens if tok != "[UNK]"]
    bad_tokens_re = "|".join(bad_tokens)
    #
    bad_cells = tf.strings.regex_full_match(token_txt, bad_tokens_re)
    result = tf.ragged.boolean_mask(token_txt, ~bad_cells)
    result = tf.strings.reduce_join(result, separator=' ', axis=-1)
    return result



if __name__ == "__main__":

  filename = os.path.join(os.path.dirname(sys.path[0]), 'log', 'vocab_token.log')
  logging.basicConfig(filename = filename,
                      level = logging.DEBUG,
                      format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')

  reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]

  """ Vocabulary """
  if 0:
    create_vocab = CreateVocab(reserved_tokens)
    create_vocab()


  # """ Tokenizer """
  if 1:
    create_tokenizer = CreateTokenizer(reserved_tokens)
    create_tokenizer()
