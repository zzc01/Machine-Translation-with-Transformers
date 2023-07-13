from pickle import dump, load
import numpy as np
from numpy.random import rand, shuffle
from unicodedata import normalize
import sys, os 
import logging 
 

class DataCleaning():
  def __call__(self):
    """
    Load data, process data, split train-val-test, and save data

    """





    """ Load the data """
    filename = os.path.join(os.path.dirname(sys.path[0]), 'data', 'deu.txt')
    with open(filename, 'rt', encoding='utf-8') as f:
        text = f.read()
    f.close()





    """ Clean: Remove 3rd Column  """
    lines = text.strip().split('\n')
    pairs = [l.split('\t')[0:2] for l in lines]

    """ Clean: Normalize and Encode to utf-8 """
    cleaned = list()
    for i, pair in enumerate(pairs):
        clean_pair = list()
        for sentence in pair:
            sentence = normalize('NFD', sentence)
            sentence = sentence.encode('utf-8')
            clean_pair.append(sentence)
        cleaned.append(clean_pair[::-1])






    """ Train-Val-Test Split"""
    n_sentence = len(cleaned)
    train_ratio = 0.8
    val_ratio   = 0.1
    test_ratio  = 0.1
    dataset = np.array(cleaned)
    shuffle(dataset)


    train = dataset[ : int(n_sentence*train_ratio)]
    val = dataset[int(n_sentence*train_ratio) : int(n_sentence*(train_ratio+val_ratio))]
    test = dataset[int(n_sentence*(train_ratio+val_ratio)) : int(n_sentence*(train_ratio+val_ratio+test_ratio))]
    
    # should actually log this information 
    logging.info(f"n_sentence = {n_sentence}")
    logging.info(f"train_ratio = {train_ratio}")
    logging.info(f"val_ratio = {val_ratio}")
    logging.info(f"test_ratio = {test_ratio}")
    logging.info(f"len(train) = {len(train)}")
    logging.info(f"len(val) = {len(val)}")
    logging.info(f"len(test) = {len(test)}")





    """ Save to txt """
    filename = os.path.join(os.path.dirname(sys.path[0]), 'data', 'deu-eng-train.txt')
    with open(filename, "w", encoding="utf-8") as output:
        for i, p in enumerate(train):
            output.write(p[0].decode('utf-8') + '\t' + p[1].decode('utf-8') + '\n')
    #
    filename = os.path.join(os.path.dirname(sys.path[0]), 'data', 'deu-eng-val.txt')
    with open(filename, "w", encoding="utf-8") as output:
        for i, p in enumerate(val):
            output.write(p[0].decode('utf-8') + '\t' + p[1].decode('utf-8') + '\n')
    #
    filename = os.path.join(os.path.dirname(sys.path[0]), 'data', 'deu-eng-test.txt')
    with open(filename, "w", encoding="utf-8") as output:
        for i, p in enumerate(test):
            output.write(p[0].decode('utf-8') + '\t' + p[1].decode('utf-8') + '\n')




    """ Save to Pkl """
    filename = os.path.join(os.path.dirname(sys.path[0]), 'data', 'deu-eng-train.pkl')
    dump(train, open(filename, 'wb'))
    #
    filename = os.path.join(os.path.dirname(sys.path[0]), 'data', 'deu-eng-val.pkl')
    dump(val, open(filename, 'wb'))
    #
    filename = os.path.join(os.path.dirname(sys.path[0]), 'data', 'deu-eng-test.pkl')
    dump(test, open(filename, 'wb'))




if __name__ == "__main__":
    data_clean = DataCleaning()
    data_clean()