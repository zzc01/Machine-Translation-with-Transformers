
import tensorflow_text
import tensorflow as tf 
import os, sys 
from matplotlib import pyplot as plt 



# def print_translation(sentence, tokens, ground_truth):
#     print(f'{"Input:":15s}: {sentence}')
#     print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
#     print(f'{"Ground truth":15s}: {ground_truth}')





# filename = os.path.join(os.path.dirname(sys.path[0]), 'metadata', 'translator')
# reloaded = tf.saved_model.load(filename)

# sentence = "Sie sind stolz darauf, Studenten jener Universität zu sein."
# ground_truth = "They are proud to be students at that university."
# translated_text, translated_tokens, attention_weights = reloaded(tf.constant(sentence))
# print_translation(sentence, translated_text, ground_truth)


loss = [
2.8969,
1.6458,
1.4011,
1.2826,
1.2049,
1.1469,
1.1031,
1.0671,
1.0375,
1.0104,
0.9879,
0.9676,
0.9481,
0.9322,
0.9156,
0.9037,
0.8907,
0.8787,
0.8672,
0.8574
]

accuracy =[
0.5447,
0.7013,
0.7398,
0.7589,
0.7719,
0.7816,
0.7894,
0.7954,
0.8007,
0.8051,
0.8093,
0.8128,
0.8164,
0.8190,
0.8220,
0.8241,
0.8267,
0.8284,
0.8307,
0.8327
]

val_loss = [
1.7704,
1.4182,
1.2962,
1.2141,
1.1739,
1.1392,
1.1144,
1.0977,
1.0779,
1.0641,
1.0519,
1.0408,
1.0336,
1.0277,
1.0243,
1.0107,
1.0080,
1.0063,
0.9991,
0.9974
]

val_accuacy = [
0.6835,
0.7390,
0.7577,
0.7732,
0.7804,
0.7876,
0.7909,
0.7949,
0.7979,
0.8010,
0.8032,
0.8050,
0.8075,
0.8075,
0.8092,
0.8118,
0.8125,
0.8136,
0.8145,
0.8146
]
