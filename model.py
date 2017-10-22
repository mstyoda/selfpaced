from tools import Progbar
from tools import mini_batches
from tools import to_onehot
import tensorflow as tf

class Model(object):
    """Abstracts a Tensorflow graph for a learning task.

    We use various Model classes as usual abstractions to encapsulate tensorflow
    computational graphs. Each algorithm you will construct in this homework will
    inherit from a Model object.
    """
    def add_placeholders(self):
        """Adds placeholder variables to tensorflow computational graph.

        Tensorflow uses placeholder variables to represent locations in a
        computational graph where data is inserted.  These placeholders are used as
        inputs by the rest of the model building and will be fed data during
        training.

        See for more information:
        https://www.tensorflow.org/versions/r0.7/api_docs/python/io_ops.html#placeholders
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def create_feed_dict(self, inputs_batch, weight_batch, labels_batch=None):
        """Creates the feed_dict for one step of training.

        A feed_dict takes the form of:
        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        If labels_batch is None, then no labels are added to feed_dict.

        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.

        Args:
            inputs_batch: A batch of input data.
            weight_batch: A batch of weight data
            labels_batch: A batch of label data.

        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_prediction_op(self):
        """Implements the core of the model that transforms a batch of input data into predictions.

        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.

        Args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A 0-d tensor (scalar) output
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        sess.run() to train the model. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Args:
            loss: Loss tensor (a scalar).
        Returns:
            train_op: The Op for training.
        """

        raise NotImplementedError("Each Model must re-implement this method.")

    def train_on_batch(self, sess, inputs_batch, weight_batch, labels_batch):
        """Perform one step of gradient descent on the provided batch of data.

        Args:
            sess: tf.Session()
            inputs_batch: np.ndarray of shape (n_samples, row_size, column_size)
            weight_batch: np.ndarray of shape(n_samples)
            labels_batch: np.ndarray of shape (n_samples, n_classes)
        Returns:
            loss: loss over the batch (a scalar)
        """
        feed = self.create_feed_dict(inputs_batch=inputs_batch, weight_batch=weight_batch, labels_batch=labels_batch)

        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)

        return loss

    def predict_on_batch(self, sess, inputs_batch):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
            inputs_batch: np.ndarray of shape (batch_size, row_size, column_size)
        Returns:
            predictions: np.ndarray of shape (batch_size, n_classes)
        """
        feed = self.create_feed_dict(inputs_batch=inputs_batch, weight_batch=None, labels_batch=None)
        predictions = sess.run(self.pred_with_softmax, feed_dict=feed)
        return predictions

    def get_accuracy(self, sess, inputs, labels):
        predictions = self.predict_on_batch(sess, inputs)
        n_predictions = len(predictions)
        accuracy = 0.0
        for i in range(0, n_predictions):
            y = 0
            correct = 1.0
            for j in range(0, len(labels[i])):
                if labels[i][j] > 0.5:
                    y = j
            for j in range(0, len(predictions[i])):
                if (y != j) and (predictions[i][j] > predictions[i][y] + 0.00001):
                    correct = 0.0
            accuracy += correct
        return accuracy / float(n_predictions)

    def run_epoch(self, sess, train_examples):
        """ Running on a epoch

        Args:
            sess: tf.Session()
            train_examples: (x, v, y)
        Return:
            epoch_loss: loss after this epoch
        """

        print self.config.batch_size
        epoch_loss = 0
        all_batches = mini_batches(train_examples, self.config.batch_size)
        prog = Progbar(target=1 + len(all_batches))

        xx, vv, yy = all_batches[0]
        print xx.dtype

        for i, (train_x, train_v, train_y) in enumerate(all_batches):
            loss = self.train_on_batch(sess=sess,
                                       inputs_batch=train_x,
                                       weight_batch=train_v,
                                       labels_batch=train_y
                                       )
            accuracy = self.get_accuracy(sess=sess, inputs=train_x, labels=train_y)
            epoch_loss = float(loss)
            prog.update(i + 1, [("train loss", loss), ("train accuracy", accuracy)])
        '''
        for i in range(0, 100):
            trainx, trainv, trainy = all_batches[0]
            loss = self.train_on_batch(sess, trainx, trainv, trainy)
            print 'i = ',i, 'loss = ',loss
        '''
        return epoch_loss

    def fit(self, sess, train_examples):
        epoch = 1
        while self.run_epoch(sess, train_examples) > 0.00001:
            print '\nAfter epoch ' + str(epoch) + 'Still not converge......'
            epoch += 1
        print 'Fit successful!'

    def get_L(self, sess, train_examples):
        """
        Args:
            sess: session of tensorflow
            train_examples: (inputs, labels)
                inputs: of shape (n_train, n_rows, n_column)
                labels: of shape (n_train)
        Return:
            L: of shape(n_train), represent the origin loss_vector of train_examples
        """

        inputs, labels = train_examples
        labels = to_onehot(labels=labels)
        weight = tf.ones(shape=labels.shape[0], dtype=tf.float32)
        feed = self.create_feed_dict(inputs_batch=inputs, weight_batch=weight, labels_batch=labels)
        L = sess.run(self.loss_vector, feed_dict=feed)
        return L

    def build(self):
        self.add_placeholders()
        self.pred, self.pred_with_softmax = self.add_prediction_op()
        self.loss, self.loss_vector = self.add_loss_op(pred=self.pred)
        self.train_op = self.add_training_op(loss=self.loss)
