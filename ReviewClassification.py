'''
@author: John Henry Dahlberg

@created: 2019-07-11

Dependencies:
Python: 3.7.3
tensorflow 1.13.1
tensorflow-hub 0.5.0

'''

import csv
from decimal import Decimal

import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
import pandas as pd

import keras.backend as K
from keras import optimizers
from keras.models import Model
from keras import layers


class ReviewrClassification(object):

    # Initialize dictionary class attributes and load data
    def __init__(self, attributes=None):
        # Check if attribute dictionary is provided
        if not attributes:
            raise Exception('Dictionary argument "attributes" is required.')

        # Set dictionary attributes to class attributes
        self.__dict__ = attributes

        # Load data from csv file and split into training, validation and testing sets
        self.train_data, self.valid_data, self.test_data = self.load_data(self.data_path)

    def load_data(self, data_path):

        # Read data
        with open(data_path, 'r', encoding="utf8") as csv_file:
            data = csv.reader(csv_file)

            train_data = []
            test_data = []

            # Iterate through each data sample
            for row in data:
                if row[-1]:
                    # Store data sample if consisting of label to then be distributed to the training and validation sets
                    train_data.append(np.flip(row[1:]))
                else:
                    # Store other samples without labels for testing to be classified
                    test_data.append(row)

            # Shuffle and split training and validation sets
            np.random.shuffle(train_data)
            valid_data = train_data[int(self.training_proportion*len(train_data)):]
            train_data = train_data[1:int(self.training_proportion*len(train_data))]

            return train_data, valid_data, test_data

    def embed_sentences(self):

        # Load universal sentence embedding model
        print('Loading encoder model...')
        self.embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")

        # Extracting training texts and labels, assign the ones later needed as class attributes i.e. self
        train_data_frame = pd.DataFrame(self.train_data, columns=['label', 'text'])
        train_data_frame.label = train_data_frame.label.astype('category')
        train_data_frame.head()
        self.train_text = train_data_frame['text'].tolist()
        self.train_text = np.array(self.train_text, dtype=object)[:, np.newaxis]
        self.train_label = np.asarray(pd.get_dummies(train_data_frame.label), dtype=np.int8)
        self.train_label = self.train_label[:, [0, -1]]

        # Extracting validation texts and labels, assign the ones later needed as class attributes i.e. self
        self.valid_data_frame = pd.DataFrame(self.valid_data, columns=['label', 'text'])
        self.valid_data_frame.label = self.valid_data_frame.label.astype('category')
        self.valid_data_frame.head()
        self.valid_text = self.valid_data_frame['text'].tolist()
        self.valid_text = np.array(self.valid_text, dtype=object)[:, np.newaxis]
        self.valid_label = np.asarray(pd.get_dummies(self.valid_data_frame.label), dtype=np.int8)

    # Perform the encoding according to the loaded embed model and make it compatible with Keras
    def universal_sentence_embedding(self, x):
        return self.embed(tf.squeeze(tf.cast(x, tf.string)),
                     signature="default", as_dict=True)["default"]

    # Create the Keras NN model
    def create_classification_model(self):

        # Takes in the texts
        input_layer = layers.Input(shape=(1,), dtype=tf.string)

        # Encodes to vectors according to the universal_sentence_embedding
        embedding_layer = layers.Lambda(self.universal_sentence_embedding,
                                  output_shape=(512,))(input_layer)

        # Point to the latest layer so that subsequent fully connected layers may be stacked
        fully_connected_layer = embedding_layer

        # Stacking fully connected laters according to 'neurons_hidden_layers'
        for n_hidden_neurons in self.neurons_hidden_layers:
            fully_connected_layer = layers.Dense(units=n_hidden_neurons, activation='relu')(fully_connected_layer)

            # Applying dropout after each layer
            fully_connected_layer = layers.Dropout(self.dropout)(fully_connected_layer)

        # Add final layer to map to the binary categories normalized according to softmax to get probabilities
        prediction_layer = layers.Dense(2, activation='softmax')(fully_connected_layer)

        # Create the model
        classification_model = Model(inputs=[input_layer], outputs=prediction_layer)

        return classification_model

    # Either perform the training or inference step, where the latter classifies texts
    def run_classifier(self):

        print('Initializing classification model...')
        classification_model = self.create_classification_model()

        if self.training:
            # Print the model architecture
            classification_model.summary()

            # Definition of the gradient decent method and loss function
            optimizer = optimizers.Adam(lr=self.learning_rate)
            classification_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

            # Perform training
            with tf.Session() as session:
                K.set_session(session)
                session.run(tf.global_variables_initializer())
                session.run(tf.tables_initializer())

                # Store info of architecture when saving Keras models
                file_info = '_epochs%d_layers%d-dropout%.2f' % (self.epochs,
                len(self.neurons_hidden_layers), self.dropout) + '-eta{:.3e}'.format(
                Decimal(self.learning_rate))

                classification_model.fit(self.train_text,
                                            self.train_label,
                                            validation_data=(self.valid_text, self.valid_label),
                                            epochs=self.epochs,
                                            batch_size=32)

                # Save the Keras model after training
                classification_model.save_weights(self.model_path + 'model_weights' + file_info + '.h5')

        else:
            # Perform inference
            print('Inference process to classify test data...')

            with tf.Session() as session:
                K.set_session(session)
                session.run(tf.global_variables_initializer())
                session.run(tf.tables_initializer())

                # Load weights of previous training
                classification_model.load_weights(self.model_weights)

                # Extracting (sorted) test texts with indices to get original order
                test_data_sorted = sorted(self.test_data, key=lambda x: int(x[0]))

                # Extracting texts
                test_text = np.array(np.array(test_data_sorted)[:, 1], dtype=object)[:, np.newaxis]

                # Perform classification
                predicts = classification_model.predict(test_text, batch_size=32)

            # Get categories
            categories = self.valid_data_frame.label.cat.categories.tolist()

            # Extracting categories with larges predicted probabilities
            predict_logits = predicts.argmax(axis=1)

            # Insert classifications in original test texts
            test_data_sorted = [test_text[i] + ', ' + categories[predict_logits[i]] for i in range(len(predict_logits))]

            # Save classifications in csv file
            with open(self.classified_path, 'w',  encoding="utf8") as writeFile:
                writer = csv.writer(writeFile)
                writer.writerows(test_data_sorted)

            print('Test data is classified and saved in "' + self.classified_path + '"')


def main():

    # Class attributes for the ReviewrClassification object
    attributes = {
        'data_path': 'Data/reviews.csv',          # Path to csv data file to train on (training=True) and complete (training=False)
        'classified_path': 'Data/classifed.csv',  # Path to save csv data file completed by classified missing labels
        'model_path': 'Data/Models/',             # Directory where to store keras models and weights
        'model_weights': 'Data/Models/model_weights_epochs7_layers1-dropout0.10-eta1.000e-3.h5',  # Directory if loading weights
        'training_proportion': 0.9,               # Proportion of (labeled) data to be used for training the rest is for validation
        'dropout': .1,                            # Dropout rate of the fully connected layers
        'epochs': 7,                              # Number of epoch iterations to train
        'learning_rate': 1e-3,                    # Learning rate for the Adams optimizer
        'training': False,                         # True for training process, False for inference process
        'neurons_hidden_layers': [256]            # Number of neurons in each hidden layer
    }

    # Initialization of ReviewrClassification class object
    reviewr_classification = ReviewrClassification(attributes)

    # Embed text data with the universial sentence encoder
    reviewr_classification.embed_sentences()

    # Create and train a model (if training=True) or perform inference process i.e. classify (training=False)
    reviewr_classification.run_classifier()


if __name__ == '__main__':
    # Seeded to keep the same results
    np.random.seed(0)
    main()
