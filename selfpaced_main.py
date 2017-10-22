from tools import *
from argminV import *
from argminW import *
from model import *
from vggModel import *
from reader import *
from math import *


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


def get_selfpaced_loss(inputs, W, v, labels, lambda_t, q_t):
    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(sess=session, save_path=W.save_path)
        train_examples = (inputs, labels)
        L = W.get_L(sess=session, train_examples=train_examples)
        tf.Session().close()

        f = (1.0 / q_t) * np.sqrt(np.sum(np.sqrt(v)))**q_t - np.sum(v)
        loss = np.dot(L, v) + lambda_t * f
    return loss


def selfpaced():

    """
        Return:
            W : the DNN model
    """
    #n_train = 10000
    #N = [n_train - 2000, n_train - 1500, n_train - 1000, n_train - 500, n_train]

    n_train = 100
    N = [n_train - 20, n_train - 15, n_train - 10, n_train - 5, n_train]

    inputs, labels = read_cifar10()
    W = argminW(inputs=inputs, v=np.ones(shape=[n_train], dtype=np.float64), labels=labels)
    maxgen = len(N)

    last_loss = 1e+10

    for t in range(0, maxgen):
        Lsort = get_Lsort(W=W, inputs=inputs, labels=labels)
        lambda_t = Lsort[N[t]]
        q_t = 2.0 * tan((1.0 - N[t] / (1.0 + N[maxgen - 1])) * pi / 2.0)
        while True:
            v = argminV(inputs=inputs, model=W, labels=labels, q_t=q_t, lambda_t=lambda_t)
            W = argminW(inputs=inputs, v=v, labels=labels)
            loss = get_selfpaced_loss(inputs, W, v, labels)
            if loss > last_loss - 1e-8:
                break
            last_loss = loss
    return W
