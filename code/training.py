"""

"""






"""# Libraries"""
import tensorflow as tf 
import numpy as np 
import tensorflow_text
from transformer_tf import Transformer
from pickle import load, dump
import sys, os
import logging 




"""# Parameters"""
n_epoch     = 20
num_layers  = 4
d_model     = 128
dff         = 512
num_heads   = 8
dropout_rate = 0.1






"""# Tokenizer"""
model_name = os.path.join(os.path.dirname(sys.path[0]), 'metadata', 'tokenizer_deu_eng')
tokenizers = tf.saved_model.load(model_name)






"""# Dataset"""
filename = os.path.join(os.path.dirname(sys.path[0]), 'data', 'deu-eng-train.pkl')
with open( filename, 'rb') as file:
    train_data = load(file)

train_data2 = np.array(train_data)
trainX = train_data2[:, 0]
trainY = train_data2[:, 1]
train_dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))

filename = os.path.join(os.path.dirname(sys.path[0]), 'data', 'deu-eng-val.pkl')
with open( filename, 'rb') as file:
    val_data = load(file)
val_data2 = np.array(val_data)
valX = val_data2[:, 0]
valY = val_data2[:, 1]
val_dataset = tf.data.Dataset.from_tensor_slices((valX, valY))

MAX_TOKENS=128
BUFFER_SIZE = 20000
BATCH_SIZE = 32

def prepare_batch(deu, eng):
    deu = tokenizers.deu.tokenize(deu)      # Output is ragged.
    deu = deu[:, :MAX_TOKENS]    # Trim to MAX_TOKENS.
    deu = deu.to_tensor()  # Convert to 0-padded dense Tensor

    eng = tokenizers.eng.tokenize(eng)
    eng = eng[:, :(MAX_TOKENS+1)]
    eng_inputs = eng[:, :-1].to_tensor()  # Drop the [END] tokens
    eng_labels = eng[:, 1:].to_tensor()   # Drop the [START] tokens

    return (deu, eng_inputs), eng_labels

def make_batches(ds):
  return (
      ds
      .shuffle(BUFFER_SIZE)
      .batch(BATCH_SIZE)
      .map(prepare_batch, tf.data.AUTOTUNE)
      .prefetch(buffer_size=tf.data.AUTOTUNE))

train_batches = make_batches(train_dataset)
val_batches = make_batches(val_dataset)






"""# Transformer"""
transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=tokenizers.deu.get_vocab_size().numpy(),
    target_vocab_size=tokenizers.eng.get_vocab_size().numpy(),
    dropout_rate=dropout_rate)





"""## Optimizer"""
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

"""## Loss and Metrics"""
def masked_loss(label, pred):
  mask = label != 0
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
  loss = loss_object(label, pred)

  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss


def masked_accuracy(label, pred):
  pred = tf.argmax(pred, axis=2)
  label = tf.cast(label, pred.dtype)
  match = label == pred

  mask = label != 0

  match = match & mask

  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match)/tf.reduce_sum(mask)






"""# Compile and Fit"""
transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy])

transformer.fit(train_batches,
                epochs=n_epoch,
                validation_data=val_batches)






"""# Inference"""
class Translator(tf.Module):
    def __init__(self, tokenizers, transformer):
        self.tokenizers = tokenizers
        self.transformer = transformer

    def __call__(self, sentence, max_length=MAX_TOKENS):
        # The input sentence is Portuguese, hence adding the `[START]` and `[END]` tokens.
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]

        sentence = self.tokenizers.deu.tokenize(sentence).to_tensor()

        encoder_input = sentence

        # As the output language is English, initialize the output with the
        # English `[START]` token.
        start_end = self.tokenizers.eng.tokenize([''])[0]
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]

        # `tf.TensorArray` is required here (instead of a Python list), so that the
        # dynamic-loop can be traced by `tf.function`.
        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions = self.transformer([encoder_input, output], training=False)

            # Select the last token from the `seq_len` dimension.
            predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.

            predicted_id = tf.argmax(predictions, axis=-1)

            # Concatenate the `predicted_id` to the output which is given to the
            # decoder as its input.
            output_array = output_array.write(i+1, predicted_id[0])

            if predicted_id == end:
                break

        output = tf.transpose(output_array.stack())
        # The output shape is `(1, tokens)`.
        text = tokenizers.eng.detokenize(output)[0]  # Shape: `()`.

        tokens = tokenizers.eng.lookup(output)[0]

        # `tf.function` prevents us from using the attention_weights that were
        # calculated on the last iteration of the loop.
        # So, recalculate them outside the loop.
        self.transformer([encoder_input, output[:,:-1]], training=False)
        attention_weights = self.transformer.decoder.last_attn_scores

        return text, tokens, attention_weights

def print_translation(sentence, tokens, ground_truth):
    logging.info(f'{"Input:":15s}: {sentence}')
    logging.info(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
    logging.info(f'{"Ground truth":15s}: {ground_truth}')


translator = Translator(tokenizers, transformer)

sentence = "Wir bekommen ein neues Auto nächsten Monat."
ground_truth = "We're getting a new car next month."
translated_text, translated_tokens, attention_weights = translator(tf.constant(sentence))
print_translation(sentence, translated_text, ground_truth)






"""# Export the model"""
class ExportTranslator(tf.Module):
    def __init__(self, translator):
        self.translator = translator

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, sentence):
        (result,
        tokens,
        attention_weights) = self.translator(sentence, max_length=MAX_TOKENS)

        return result, tokens, attention_weights


translator = ExportTranslator(translator)
filename = os.path.join(os.path.dirname(sys.path[0]), 'metadata', 'translator')
tf.saved_model.save(translator, export_dir=filename)

sentence = "Sie sind stolz darauf, Studenten jener Universität zu sein."
ground_truth = "They are proud to be students at that university."
translated_text, translated_tokens, attention_weights = translator(tf.constant(sentence))
print_translation(sentence, translated_text, ground_truth)


reloaded = tf.saved_model.load(filename)

sentence = "Sie sind stolz darauf, Studenten jener Universität zu sein."
ground_truth = "They are proud to be students at that university."
translated_text, translated_tokens, attention_weights = reloaded(tf.constant(sentence))
print_translation(sentence, translated_text, ground_truth)

