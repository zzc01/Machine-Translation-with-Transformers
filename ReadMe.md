# Machine Translation with Transformers 

Here is an implmentation of the machine translator using the encoder-decoder architecture with transformers. The translator is trained to translate German sentences to English sentences. <br/>


1. The data set used here is the English-German sentence pair from http://www.manythings.org/anki/

The data is cleaned and split into train, validation, and testing sets. The script used to clean the data is in [data_cleaning.ipynb](/Translator_Transformer/data_cleaning.ipynb). Here we removed unused column from the original data, applied canonical decomposition normalization to each sentence word, and encoded the sentence to Bytes datastructure using utf-8. 

2. Sub-word WordPiece tokenizer

Traditional Word-based tokenizer suffers from very large vocabularies, out-of-vocailary tokens, and loss of meaning across similar words. Sub-word tokenizers solves this by decomposing rare words into smaller meaningful subwords. In [tokenizer_tf_deu_eng.ipynb](/Translator_Transformer/tokenizer_tf_deu_eng.ipynb) we first create a vocabulary list using bert_vocab.bert_vocab_from_dataset(). Next we build the tokenizer with tensorflow_text.BertTokenizer(). One tokenizer is built for german words and second tokenizer for english words. Finaly the tokenizers are saved for later use. 

3. Here we use the Encoder-Decoder Model with Transformer neural network for the German to English translation task. 

The transformer model contain different building parts, including the mulit-head attention block, encoder, decoder, and positional encoder. Here is a block diagram of the transformer [1]. 

<pre><p align="center">
<img src="https://github.com/zzc01/Transformer/assets/86133411/b05f651e-2c70-4d2c-a164-be27b1f89e3b"  width="300" >
</p></pre>

6. Training the model 

The transfomer model and the training of the model are both implemented in [train_tf.ipynb](/Translator_Transformer/train_tf.ipynb). Here we use an adam optimizer with custom learning rate scheduler. The masked loss is calculated using SparseCategoricalCrossentropy and padding mask.  

7. Inference

Below shows the translation result and attention plot of a German sentence "Fass nichts an, ohne zu fragen!" to English sentence. </br>

**Input:         : Fass nichts an, ohne zu fragen!</br>**
**Prediction     : don ' t touch anything without asking .</br>**
**Ground truth   : Don't touch anything without asking.</br>**

<pre><p align="center">
<img src="https://github.com/zzc01/Transformer/assets/86133411/dad5085a-9165-40a4-a815-bd62f89be952"  width="400" >
</p></pre>


8. The model is evluated using BLEU score in [evaluate.ipynb](/Translator_Transformer/evaluate.ipynb). Below shows the BLEU score result. 

The left BLEU scores are the result of scoring the prediction against the raw target sentence. The right BLEU scores are scoring the prediction against the tokenized and then detokenized raw target sentence. The reason to do the tokenized and detokenized step is to seperate the puntuations and the words. For example to convert "don't" to "don ' t", and "the end." to "the end ."

<pre><p align="center">
<img src="https://github.com/zzc01/Transformer/assets/86133411/4d29b61e-4753-498c-939b-694859b67b5c"  width="400" >
</p></pre>

9. The code is referenced from Tensorflow Tutorials [2] and Minchine Learning Mastery [3]


# References 
[1] [Tensorflow Tutorials](https://www.tensorflow.org/text/tutorials) <br/>
[2] [Machine Learning Mastery](https://machinelearningmastery.com/) <br/> 
[3] [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)<br/>
[4] [Hugging Face](https://huggingface.co/learn/nlp-course/chapter0/1?fw=pt) <br/>
