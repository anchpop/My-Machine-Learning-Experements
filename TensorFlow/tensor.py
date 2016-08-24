# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 08:39:10 2016

@author: anchpop
"""
import tensorflow as tf
import numpy as np

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TRAINING, target_dtype=np.int)
test_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TEST, target_dtype=np.int)

# Set better names for the data
x_train, x_test, y_train, y_test = training_set.data, test_set.data, training_set.target, test_set.target

# Build 3 layer DNN with 10, 20, 10 units respectively. 
classifier = tf.contrib.learn.DNNClassifier(hidden_units=[3, 9, 3], n_classes=3)

# Fit model
classifier.fit(x=x_train, y=y_train, steps=200)

# Print how accurate our net is
accuracy_score = classifier.evaluate(x=x_test, y=y_test)["accuracy"]
print(('-'*10 + '\nAccuracy: {0:f}\n' + '-'*10).format(accuracy_score))