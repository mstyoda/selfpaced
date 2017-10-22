from tools import *
from argminV import *
from argminW import *
from model import *
from vggModel import *
from reader import *


def get_Lsort(W, inputs, labels):
    """
    Args:
        W: DNN models
        inputs: inputs of shape(n_train, n_rows, n_columns)
        labels: labels of shape(n_train)
    Return:
        Lsort : sorted L from small to large
    """
    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(sess=session, save_path=W.save_path)
        train_examples = (inputs, labels)
        L = W.get_L(sess=session, train_examples=train_examples)
        tf.Session().close()
        L = sorted(L)
    return L


def selfpaced():
    """
        Return:
            W : the DNN model
    """
    n_train = 10000
    N = [n_train - 2000, n_train - 1500, n_train - 1000, n_train - 500, n_train]
    inputs, labels= read_cifar10()
    W = argminW(inputs=inputs, v=np.ones(shape=[n_train], dtype=np.float32), labels=labels)
    for t in range(0,len(N)):
        lambda_t = get_Lsort(W=W, inputs=inputs, labels=labels)
