from vggModel import *
from cnnModel import *
from reader import *
import numpy as np


def argminV(inputs, model, labels, q_t, lambda_t):
    """
    Args:
        inputs: np.ndarray of shape(n_train, row_size, column_size)
        model: structure of DNN model (like vggModel or cnnModel)
        labels: np.ndarray of shape(n_train)
        q_t: q(t) in t'th iteration(a float constant)
        lambda_t: lambda^t(a float constant)
    Return:
        V: a float vector of size (n_train)
    """
    with tf.Graph().as_default():
        with tf.Session() as session:
            print "Loading model...",
            start = time.time()
            W = VggModel(Config)
            saver = tf.train.Saver()
            saver.restore(sess=session, save_path=W.save_path)
            print "took {:.2f} seconds\n".format(time.time() - start)
            train_examples = (inputs, labels)

            n_train = len(labels)
            L = W.get_L(sess=session, train_examples=train_examples)
            tf.Session().close()

            V = []
            for i in range(0,n_train):
                if L[i] < lambda_t - 1e-5:
                    V.append((1.0 - L[i] / lambda_t)**(1.0/(q_t - 1.0)))
                else:
                    V.append(0.0)
            V = np.array(V)
    return V
