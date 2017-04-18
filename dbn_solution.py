from __future__ import print_function
from __future__ import division
from dbn.tensorflow import SupervisedDBNClassification
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
train_var = 8000
dbn = SupervisedDBNClassification(
    hidden_layers_structure=[1024, 1024],
    learning_rate_rbm=0.1,
    learning_rate=0.1,
    n_epochs_rbm=10,
    n_iter_backprop=100,
    batch_size=100,
    activation_function='sigmoid',
    dropout_p=0.2
)

"""Load pickle file"""
def unpickle(filename):
    
    import pickle
    with open(filename, 'rb') as source:
        value_dict = pickle.load(source, encoding='bytes')
    return value_dict
def get_data():
    """
    Unpickle the data.
    """
    temp = unpickle("CIFAR-3.pickle")
    labels = []
    for index in range(len(temp['y'])):
        if temp['y'][index, 0] == 1:
            labels.append(1)                      #airplane image
        elif temp['y'][index, 1] == 1:            #dog image
            labels.append(2)
        else:                                     #boat image
            labels.append(3)
    """
    Train data
    """
    x_train = temp['x'][:train_var]
    x_train /= 255
    y_train = labels[:train_var]
	"""
    Test data
    """
    x_test = temp['x'][train_var:]
    x_test /= 255
    y_test = labels[train_var:]
    return x_train, y_train, x_test, y_test
x_train, y_train, x_test, y_test = get_data()
dbn.fit(x_train, y_train)
predictions = dbn.predict(x_test)
accuracy = accuracy_score(y_test, list(predictions))
print('Accuracy: {0}'.format(accuracy))