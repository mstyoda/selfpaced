from vggModel import *
from cnnModel import *
from reader import *

def argminW(inputs, v, labels):
    """
    Args:
        v: weight vector defined in "Self-paced Convolutional Neural Networks"
        inputs: np.ndarray of shape(n_train, row_size, column_size)
        labels: np.ndarray of shape(n_train)
    Return: Nothing but save DNN model weight in "data/$Name$_model.weights"
    """
    with tf.Graph().as_default():
        print "Building model...",
        start = time.time()
        W = VggModel(Config)
        print "took {:.2f} seconds\n".format(time.time() - start)

        train_examples = (inputs, v, labels)

        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)

            print 80 * "="
            print "TRAINING"
            print 80 * "="

            W.fit(session, train_examples)
            saver.save(sess=session, save_path=W.save_path)
        tf.Session().close()

if __name__ == '__main__':
    (x, y) = read_cifar10()
    v = np.ones(y.size).astype('float32')
    argminW(x, v, y)
