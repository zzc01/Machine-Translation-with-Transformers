import tensorflow as tf
import tensorflow_text as tftxt
from pickle import load, dump
from matplotlib import pyplot as plt 
import numpy as np

import sys, os
print(sys.path[0])
print(os.path.join(os.path.dirname(sys.path[0]), 'code'))
sys.path.insert(0, os.path.join(os.path.dirname(sys.path[0]), 'code'))
import logging 
filename = os.path.join(os.path.dirname(sys.path[0]), 'log', 'test.log')
logging.basicConfig(filename = filename,
                    level = logging.DEBUG,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')
# logging.debug('Debug message')
# logging.info('Info message')
# logging.warning('Warning message')
# logging.error('Error message')
# logging.critical('Critical message')

from data_cleaning import DataCleaning
from tokenizer_tf import CreateVocab, CreateTokenizer 
from positional_encoding_tf import positional_encoding, PositionalEmbedding 
from attention_tf import GlobalSelfAttention, CausalSelfAttention, CrossAttention
from encoder_tf import EncoderLayer, FeedForward, Encoder 
from decoder_tf import DecoderLayer, Decoder 
from transformer_tf import Transformer 
from train_tb_tf import CustomSchedule




"""
Data Set Test 
    Make sure data set exists 
    Make sure the data set shape is correct 2 column and more than 200K rows 
        Can use old hand checked data to check 
    Make sure it is Deu and Eng
"""
def data_clean_tb():

    # Run the data cleaning script 
    data_cleaning = DataCleaning()
    data_cleaning()
   

    # Load data set from Pkl to check 
    filename = os.path.join(os.path.dirname(sys.path[0]), 'data', 'deu-eng-train.pkl')
    with open(filename, 'rb') as file:
        dataset = load(file)
    logging.info(f"Shape of deu-eng-train.pkl = {dataset.shape}")
    
    # If the length is too short. The data set is incorrect.     
    if dataset.shape[0] > 2e5 :
        logging.info("The length of deu-eng-train.pkl is normal greater than 2e5")
    else:
        logging.info("The length of deu-eng-train.pkl is too short")
    for i, pair in enumerate(dataset):
        if i == 3: break
        logging.info([pair[0], pair[1]])

    # Can continue to check the shape of val, test 





"""
Tokenzier Tests 
    Check Vocabulary 
    Make sure vocabulary txt exists 
    Make sure shape is correct
    Make sure check if it is Deu and Eng
    can use old hand checked data to check 
    Make sure tokenizer works 
"""
def create_vocab_tb():
    reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]
    # create_vocab = CreateVocab(reserved_tokens)
    # create_vocab()

    # Load deu_vocab 
    filename = os.path.join(os.path.dirname(sys.path[0]), 'metadata', 'deu_vocab.txt')
    with open(filename, 'rt', encoding="utf-8") as file:
        vocab = file.read()
    logging.info(f"length of due_vocab = {len(vocab.split())}")
    logging.info(f"deu_vocab[:10] = {vocab.split()[:10]}")
    logging.info(f"deu_vocab[100:110] = {vocab.split()[100:110]}")
    logging.info(f"deu_vocab[1000:1010] = {vocab.split()[1000:1010]}")
    logging.info(f"deu_vocab[-10:] = {vocab.split()[-10:]}")

    # If the length is too short. The vocab is incorrect.     
    if len(vocab.split()) > 7e3 :
        logging.info("The length of deu_vocab is normal greater than 7e3")
    else:
        logging.info("The length of deu_vocab is too short")

    # Load eng_vocab 
    filename = os.path.join(os.path.dirname(sys.path[0]), 'metadata', 'eng_vocab.txt')
    with open(filename, 'rt', encoding="utf-8") as file:
        vocab = file.read()
    logging.info(f"length of eng_vocab = {len(vocab.split())}")
    logging.info(f"eng_vocab[:10] = {vocab.split()[:10]}")
    logging.info(f"eng_vocab[100:110] = {vocab.split()[100:110]}")
    logging.info(f"eng_vocab[1000:1010] = {vocab.split()[1000:1010]}")
    logging.info(f"eng_vocab[-10:] = {vocab.split()[-10:]}")

    # If the length is too short. The vocab is incorrect.     
    if len(vocab.split()) > 5e3 :
        logging.info("The length of eng_vocab is normal greater than 5e3")
    else:
        logging.info("The length of eng_vocab is too short")

def create_tokenizer_tb():
    reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]
    create_tokenizer = CreateTokenizer(reserved_tokens)
    create_tokenizer()

    # Load and Test The Model German
    model_name = os.path.join(os.path.dirname(sys.path[0]), 'metadata', 'tokenizer_deu_eng')
    reloaded_tokenizers = tf.saved_model.load(model_name)
    logging.info(f"vocab lenght = {reloaded_tokenizers.deu.get_vocab_size().numpy()}")
    # 
    tokens = reloaded_tokenizers.deu.tokenize(['Ich habe mein Geld für Kleidung, Essen und Bücher ausgegeben.'])
    logging.info("Ich habe mein Geld für Kleidung, Essen und Bücher ausgegeben.")
    logging.info(tokens.numpy())
    #
    text_tokens = reloaded_tokenizers.deu.lookup(tokens)
    logging.info(text_tokens)
    #
    round_trip = reloaded_tokenizers.deu.detokenize(tokens)
    logging.info(round_trip.numpy()[0].decode('utf-8'))

    # Load and Test The Model English
    string = "When writing a sentence, generally you start with a capital letter and finish with a period (.), an exclamation mark (!), or a question mark (?)."
    tokens = reloaded_tokenizers.eng.tokenize([string])
    logging.info(string)
    logging.info(tokens.numpy())
    text_tokens = reloaded_tokenizers.eng.lookup(tokens)
    logging.info(text_tokens)
    round_trip = reloaded_tokenizers.eng.detokenize(tokens)
    logging.info(round_trip.numpy()[0].decode('utf-8'))





"""
Positional Encoder and Word Embedding Tests 
"""
def positional_encoding_tb():
    
    pos_encoding = positional_encoding(length=2048, depth=512)

    # Check the shape.
    logging.info("positional encoding input length = 2048, depth = 512")
    logging.info(f"positional encoding output shape = {pos_encoding.shape}")
    
    # Plot the dimensions.
    plt.pcolormesh(pos_encoding.numpy().T, cmap='RdBu')
    plt.ylabel('Depth')
    plt.xlabel('Position')
    plt.colorbar()
    plt.show()

    pos_encoding /= tf.norm(pos_encoding, axis=1, keepdims=True)
    p = pos_encoding[1000]
    dots = tf.einsum('pd,d -> p', pos_encoding, p)
    plt.subplot(2,1,1)
    plt.plot(dots)
    plt.ylim([0,1])
    plt.plot([950, 950, float('nan'), 1050, 1050],
             [0,1,float('nan'),0,1], color='k', label='Zoom')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(dots)
    plt.xlim([950, 1050])
    plt.ylim([0,1])
    plt.show()


def positional_embedding_tb():

    logging.info(f"Positional Embedding Test Bench Start")

    """ Load Data Set """
    filename = os.path.join(os.path.dirname(sys.path[0]), 'data', 'deu-eng-train.pkl')
    with open(filename, 'rb') as file:
      dataset = load(file)

    trainX = np.array(dataset)[:, 0]
    trainY = np.array(dataset)[:, 1]
    train_dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))

    """ Load Tokenizer """
    model_name = os.path.join(os.path.dirname(sys.path[0]), 'metadata', 'tokenizer_deu_eng')
    reloaded_tokenizers = tf.saved_model.load(model_name)

    embed_deu = PositionalEmbedding(vocab_size=reloaded_tokenizers.deu.get_vocab_size(), d_model=512)
    embed_eng = PositionalEmbedding(vocab_size=reloaded_tokenizers.eng.get_vocab_size(), d_model=512)

    """ Batch Parameters """
    MAX_TOKENS  = 128
    BUFFER_SIZE = 20000
    BATCH_SIZE  = 32

    def prepare_batch(deu, eng):
        deu = reloaded_tokenizers.deu.tokenize(deu)      # Output is ragged.
        deu = deu[:, :MAX_TOKENS]    # Trim to MAX_TOKENS.
        deu = deu.to_tensor()  # Convert to 0-padded dense Tensor

        eng = reloaded_tokenizers.eng.tokenize(eng)
        eng = eng[:, :(MAX_TOKENS+1)]
        eng_inputs = eng[:, :-1].to_tensor()  # Drop the [END] tokens
        eng_labels = eng[:, 1:].to_tensor()   # Drop the [START] tokens

        return (deu, eng_inputs), eng_labels

    def make_batches(ds):
        return (
            ds
            # .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .map(prepare_batch, tf.data.AUTOTUNE)
            .prefetch(buffer_size=tf.data.AUTOTUNE))

    train_batches = make_batches(train_dataset)
    for (deu, eng), eng_labels in train_batches.take(1):
        break

    logging.info(f"deu batch shape = {deu.shape}")
    logging.info(f"eng batch shape = {eng.shape}")
    logging.info(f"eng labels batchshape = {eng_labels.shape}")
    logging.info(f"A eng sentence = {eng[0][:10]}")
    logging.info(f"A eng_labels sentence = {eng_labels[0][:10]}")

    deu_emb = embed_deu(deu)
    eng_emb = embed_eng(eng)

    logging.info(f"deu shape after embedding = {deu_emb.shape}")
    logging.info(f"eng shape after embedding = {eng_emb.shape}")
    logging.info(f"eng.shape = {eng.shape}")
    logging.info(f"eng[0:3] = {eng[0:3]}")
    logging.info(f"eng_emb._keras_mask.shape = {eng_emb._keras_mask.shape}")
    logging.info(f"eng_emb._keras_mask[0:3] = {eng_emb._keras_mask[0:3]}")

    logging.info(f"Positional Embedding Test Bench Finish")





"""
Multihead Attention Tests
"""

# """
# create an known input and output pair 
# """
def attention_tb():

    logging.info(f"Attention Test Bench Start")

    """ Load Data Set """
    filename = os.path.join(os.path.dirname(sys.path[0]), 'data', 'deu-eng-train.pkl')
    with open(filename, 'rb') as file:
      dataset = load(file)

    trainX = np.array(dataset)[:, 0]
    trainY = np.array(dataset)[:, 1]
    train_dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))

    """ Load Tokenizer """
    model_name = os.path.join(os.path.dirname(sys.path[0]), 'metadata', 'tokenizer_deu_eng')
    reloaded_tokenizers = tf.saved_model.load(model_name)

    embed_deu = PositionalEmbedding(vocab_size=reloaded_tokenizers.deu.get_vocab_size(), d_model=512)
    embed_eng = PositionalEmbedding(vocab_size=reloaded_tokenizers.eng.get_vocab_size(), d_model=512)

    """ Batch Parameters """
    MAX_TOKENS  = 128
    BUFFER_SIZE = 20000
    BATCH_SIZE  = 32

    def prepare_batch(deu, eng):
        deu = reloaded_tokenizers.deu.tokenize(deu)      # Output is ragged.
        deu = deu[:, :MAX_TOKENS]    # Trim to MAX_TOKENS.
        deu = deu.to_tensor()  # Convert to 0-padded dense Tensor

        eng = reloaded_tokenizers.eng.tokenize(eng)
        eng = eng[:, :(MAX_TOKENS+1)]
        eng_inputs = eng[:, :-1].to_tensor()  # Drop the [END] tokens
        eng_labels = eng[:, 1:].to_tensor()   # Drop the [START] tokens

        return (deu, eng_inputs), eng_labels

    def make_batches(ds):
        return (
            ds
            # .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .map(prepare_batch, tf.data.AUTOTUNE)
            .prefetch(buffer_size=tf.data.AUTOTUNE))

    train_batches = make_batches(train_dataset)
    for (deu, eng), eng_labels in train_batches.take(1):
        break

    logging.info(f"deu batch shape = {deu.shape}")
    logging.info(f"eng batch shape = {eng.shape}")
    logging.info(f"eng labels batchshape = {eng_labels.shape}")
    logging.info(f"A eng sentence = {eng[0][:10]}")
    logging.info(f"A eng_labels sentence is just a left shift= {eng_labels[0][:10]}")

    deu_emb = embed_deu(deu)
    eng_emb = embed_eng(eng)

    logging.info(f"deu shape after embedding = {deu_emb.shape}")
    logging.info(f"eng shape after embedding = {eng_emb.shape}")
    # print(en_emb._keras_mask)


    #def GlobalSelfAttention_tb():
    sample_gsa = GlobalSelfAttention(num_heads=2, key_dim=512)
    logging.info(f"Shape before global self attention = {deu_emb.shape}")
    logging.info(f"Shape after global self attention = {sample_gsa(deu_emb).shape}")

    # def CrossAttention_tb():
    """The output length of CrossAttention is the length of the query sequence, 
    not the length of the context key/value sequence.
    """
    sample_ca = CrossAttention(num_heads=2, key_dim=512)
    logging.info(f"X shape before cross attention = {eng_emb.shape}")
    logging.info(f"Context shape before cross attention = {deu_emb.shape}")
    logging.info(f"Shape after cross attention = {sample_ca(eng_emb, deu_emb).shape}")

    # Check the shap of CA output  
    if sample_ca(eng_emb, deu_emb).shape == eng_emb.shape:
        logging.info("The CA output shape is normal.")
    else:
        logging.info("The CA output shape is worn.")

    # def CausalSelfAttention_tb():
    sample_csa = CausalSelfAttention(num_heads=2, key_dim=512)
    logging.info(f"Shape before causal self attention = {deu_emb.shape}")
    logging.info(f"Shape after causal self attention = {sample_csa(deu_emb).shape}")

    """To test the causality of CSA
    Trancating the sentence before CSA or after CSA, the result
    should be the same because of the causal mask. 

    """

    out1 = sample_csa(embed_eng(eng[:, :3]))
    out2 = sample_csa(embed_eng(eng))[:, :3]

    logging.info(f"out1.shape = {out1.shape}")
    logging.info(f"out2.shape = {out2.shape}")

    # check the difference between two CSA result 
    if tf.reduce_max(abs(out1 - out2)).numpy() < 0.1 :
        logging.info("The CSA difference is normal < 0.1")
    else:
        logging.info("The CSA difference is too large greater than 0.1")



    logging.info(f"Attention Test Bench Finish")





"""
Encoder Tests
"""
def feedforward_tb():
    logging.info(f"FeedForward Test Bench Start")
    eng_emb = tf.random.normal([32, 21, 512], dtype=tf.float64)
    sample_ffn = FeedForward(512, 2048)
    logging.info(eng_emb.shape)
    logging.info(sample_ffn(eng_emb).shape)
    logging.info(f"FeedForward Test Bench Finish")

def encoderlayer_tb():
    logging.info(f"EncoderLayer Test Bench Start")
    eng_emb = tf.random.normal([32, 21, 512], dtype=tf.float64)
    sample_encoder_layer = EncoderLayer(d_model=512, num_heads=8, dff=2048)
    logging.info(eng_emb.shape)
    logging.info(sample_encoder_layer(eng_emb).shape)
    logging.info(f"EncoderLayer Test Bench Finish")

def encoder_tb():
    logging.info(f"Encoder Test Bench Start")
    eng = tf.random.normal([32, 21], dtype=tf.float64)
    # Instantiate the encoder.
    sample_encoder = Encoder(num_layers=4,
                            d_model=512,
                            num_heads=8,
                            dff=2048,
                            vocab_size=8500)

    sample_encoder_output = sample_encoder(eng, training=False)

    # Print the shape.
    logging.info(eng.shape)
    logging.info(sample_encoder_output.shape)  # Shape `(batch_size, input_seq_len, d_model)`.
    logging.info(f"Encoder Test Bench Finish")






"""
Dencoder Tests
"""
def decoderlayer_tb():
    logging.info(f"Decoder Layer Test Bench Start")
    eng_emb = tf.random.normal([32, 21, 512], dtype=tf.float64)
    deu_emb = tf.random.normal([32, 19, 512], dtype=tf.float64)

    sample_decoder_layer = DecoderLayer(d_model=512, num_heads=8, dff=2048)
    sample_decoder_layer_output = sample_decoder_layer(
        x=eng_emb, context=deu_emb)

    logging.info(eng_emb.shape)
    logging.info(deu_emb.shape)
    logging.info(sample_decoder_layer_output.shape)  # `(batch_size, seq_len, d_model)`
    logging.info(f"Decoder Layer Test Bench Finish")

def decoder_tb():

    logging.info(f"Decoder Test Bench Start")
    eng = tf.random.normal([32, 21], dtype=tf.float64)
    deu_emb = tf.random.normal([32, 19, 512], dtype=tf.float64)
    # Instantiate the decoder.
    sample_decoder = Decoder(num_layers=4,
                            d_model=512,
                            num_heads=8,
                            dff=2048,
                            vocab_size=8000)

    output = sample_decoder(
        x=eng,
        context=deu_emb)


    # Print the shapes.
    logging.info(eng.shape)
    logging.info(deu_emb.shape)
    logging.info(output.shape)
    logging.info(sample_decoder.last_attn_scores.shape)
    logging.info(f"Decoder Test Bench Finish")



"""
Transformer Tests
"""

def transformer_tb():
    logging.info(f"Transformer Test Bench Start")
    eng = tf.random.normal([32, 21], dtype=tf.float64)
    deu = tf.random.normal([32, 19], dtype=tf.float64)

    """# Parameters"""
    num_layers  = 4
    d_model     = 128
    dff         = 512
    num_heads   = 8
    dropout_rate = 0.1
    # Instantiate the decoder.
    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        # input_vocab_size=tokenizers.deu.get_vocab_size().numpy(),
        # target_vocab_size=tokenizers.eng.get_vocab_size().numpy(),
        input_vocab_size=8000,
        target_vocab_size=7000,
        dropout_rate=dropout_rate)


    output = transformer((deu, eng))

    logging.info(eng.shape)
    logging.info(deu.shape)
    logging.info(output.shape)

    """See the head actually multiplies the input matrix. Because it acutally duplicates the matrix. But in Mastery it reshapes the input matrix which is wierd.** <br>
    In the Mastery earlier introduction to multi-head attention it say it duplicates the matricies by number of header does the same scaled dot product attention. And then concate the #heads and go through a dense to combine them.** <br>

    """

    attn_scores = transformer.decoder.dec_layers[-1].last_attn_scores
    logging.info(attn_scores.shape)  # (batch, heads, target_seq, input_seq)
    logging.info(transformer.summary())

    logging.info(f"Transformer Test Bench Finish")


def custom_schedule_tb():
    d_model = 512 
    learning_rate = CustomSchedule(d_model)
    plt.plot(learning_rate(tf.range(40000, dtype=tf.float32)))
    plt.ylabel('Learning Rate')
    plt.xlabel('Train Step')




if __name__ == "__main__":
    if 0: data_clean_tb()
    if 0: create_vocab_tb()
    if 0: create_tokenizer_tb()
    #
    if 0: positional_encoding_tb()
    if 0: positional_embedding_tb()
    #
    if 0: attention_tb()
    #
    if 0: feedforward_tb()
    if 0: encoderlayer_tb()
    if 0: encoder_tb() 
    #
    if 0: decoderlayer_tb()
    if 0: decoder_tb()
    #
    if 0: transformer_tb()
    #
    if 1: custom_schedule_tb()