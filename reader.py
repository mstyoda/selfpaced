import cPickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


def read_cifar10(dirname='cifar10'):
    train_dict = {}
    for i in range(1, 6):
        dict = unpickle(dirname + '/data_batch_' + str(i))
        train_dict.update(dict)

    inputs = np.reshape(np.array(train_dict['data']), (-1, 32, 32, 3)).astype('float64')
    inputs = (inputs - np.mean(inputs)) / 255.0
    labels = np.array(train_dict['labels']).astype('int32')
    return inputs[:10000], labels[:10000]

if __name__ == '__main__':
    read_cifar10()

