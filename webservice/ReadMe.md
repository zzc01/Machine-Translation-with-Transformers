# Machine Translation with Transformers 

Here is an implmentation of the machine translator using the encoder-decoder architecture with transformers [1]. The translator is trained to translate German sentences to English sentences. The code is referenced from Tensorflow Tutorials [2] and Minchine Learning Mastery [3] <br/>

# Data set 

The data set used here is the English-German sentence pair from http://www.manythings.org/anki/

The data is cleaned and split into train, validation, and testing sets. The script used to clean the data is in [data_cleaning.py](/code/data_cleaning.py). Here we removed unused column from the original data, applied canonical decomposition normalization to each sentence word, and encoded the sentence to Bytes datastructure using utf-8. 

# Sub-word WordPiece tokenizer

Tokenization is the process of breaking up text, into "tokens". Depending on the tokenizer, these tokens can represent sentence-pieces, words, subwords, or characters. Traditional Word-based tokenizer suffers from very large vocabularies, out-of-vocailary tokens, and loss of meaning across similar words. Sub-word tokenizers solves this by decomposing rare words into smaller meaningful subwords [4]. In [tokenizer_tf.py](/code/tokenizer_tf.py) we first create a vocabulary list using bert_vocab.bert_vocab_from_dataset(). Next we build the tokenizer with tensorflow_text.BertTokenizer(). One tokenizer is built for german words and second tokenizer for english words. Finaly the tokenizers are saved for later use. 

# The encoder-decoder model with neural network fransformer  

The transformer model contain different building parts, including the mulit-head attention block, encoder, decoder, and positional encoder. Here is a block diagram of the transformer [1]. 

<pre><p align="center">
<img src="https://github.com/zzc01/Transformer/assets/86133411/b05f651e-2c70-4d2c-a164-be27b1f89e3b"  width="300" >
</p></pre>

## Multihead attention 

Here shows the block diagrams of scaled dot-product attention and multi-head attention [2][3].

<pre><p align="center">
<img src="https://github.com/zzc01/Transformer/assets/86133411/2662ac99-ee81-418f-af30-3d0eaf6e560e"  width="450" >
</p></pre>

The implementation of multi-headattention includes a linear layer, reshape, and transpose as shown here.  

<pre><p align="center">
<img src="https://github.com/zzc01/Transformer/assets/86133411/752a0cb9-5b94-4c40-a767-641f0b575e36"  width="200" >
</p></pre>

# Training  

The transfomer model and the training of the model are both implemented in [training.py](/code/training.py). Here we use an adam optimizer with custom learning rate scheduler. The masked loss is calculated using SparseCategoricalCrossentropy and padding mask.  

<pre><p align="center">
<img src="https://github.com/zzc01/Machine-Translation-with-Transformers/assets/86133411/cbecb9bb-be30-4d5d-b8d1-167433e53749"  width="400" >
</p></pre>

# Inference and Attention Plot

Below shows the translation result and attention plot of a German sentence "Fass nichts an, ohne zu fragen!" to English sentence. </br>

**Input:         : Fass nichts an, ohne zu fragen!</br>**
**Prediction     : don ' t touch anything without asking .</br>**
**Ground truth   : Don't touch anything without asking.</br>**

<pre><p align="center">
<img src="https://github.com/zzc01/Transformer/assets/86133411/dad5085a-9165-40a4-a815-bd62f89be952"  width="400" >
</p></pre>

# Evaluation 

The model is evluated using BLEU score in [evaluate.py](/code/evaluate.py). Below shows the BLEU score result. The BLEU scores on the left are the result of scoring the prediction against the raw target sentence. The right BLEU scores are scoring the prediction against the tokenized and then detokenized raw target sentence. The reason to do the tokenized and detokenized step is to seperate the puntuations and the words. For example to convert "don't" to "don ' t", and "the end." to "the end ."

<pre><p align="center">
<img src="https://github.com/zzc01/Transformer/assets/86133411/4d29b61e-4753-498c-939b-694859b67b5c"  width="400" >
</p></pre>



# References 
[1] [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)<br/>
[2] [Tensorflow Tutorials](https://www.tensorflow.org/text/tutorials) <br/>
[3] [Machine Learning Mastery](https://machinelearningmastery.com/) <br/> 
[4] [Hugging Face](https://huggingface.co/learn/nlp-course/chapter0/1?fw=pt) <br/>
