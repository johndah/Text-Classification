# Text-Classification

## Problem
The problem is to classify the review paragraphs in `Data/reviews` that have missing labels, such as in line 185:
`"This item does what it's supposed to do, and it does it well.  I am very satisfied with this purchase.  Being a college student, I don't want to bother my roommate while I'm watching a show or movie with my boyfriend, so we use the headphone splitter to be quiet.",` with either `software` or `hardware`.


## Solution

The main steps to solve the problem in this implementation consist of the
* Training step, where a model is needed for the
  * Embedding, using the universal sentence encoder,
  * Neural Network architecture,
* Inference step.

The embedding is based on the paper [Attention is All you Need](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) by Vaswani et al. Nevertheless, only the encoder part is needed from the encoder-decoder architecture. The Neural Network architecture consists of
* an embedding layer, 
* a hidden, fully connected, layer with 'ReLu' activation,
* a fully connected output layer with softmax activation

The [large trained model](https://tfhub.dev/google/universal-sentence-encoder-large/3) is utilized for this implementation. This model is then used in the initial, embedding, layer of the Neural Network, to transform text paragraphs into vectors of dimentionality 512. This is follwed by the hidden fully connected layer where dropout is applied with a rate of 0.1 to avoid overfitting. A ReLu activation is also used in this layer. Then, the output layer is added with two neurons to match the categorical one-hot output, with a softmax activation in order to normalize the values to represent probabilities.

The optimizer is Adam with Keras default learning rate 0.001 along with the other default parameters. Furthermore, the `categorical crossentropy` is used as loss function. No early stopping is used. Moreover, testing of multiple hyper-parameters such as randomized search was not performed.

The data is split into a training, validation and a test set, where the latter simply corresponds to all text samples without any label. The labeled data is shuffled and 90% is used for the training set, while the rest is for the validation set.

Finally, the inference step involves classifying the test data. Saved model weights are loaded and prediction is performed where the argmax function is used for classifying texts with labels of highest probability. The classified test set is saved in its own in a new csv file, with the original order, to give an adequate overview for manual validation (see attached picture.). A final validation accuracy after 7 epochs of 99.22% was achieved.
