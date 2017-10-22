from vggModel import *
import tensorflow as tf
import time
import numpy as np
import sys


def normal_init():
    def _normal_initializer(name, shape, **kwargs):
        initializer = tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)
    return _normal_initializer


def conv2D(x, filter, bias):
    return tf.nn.relu(
                tf.nn.bias_add(
                    tf.nn.conv2d(x, filter, [1, 1, 1, 1], padding='SAME'),
                    bias
                )
    )


def maxpool(x):
    return tf.nn.lrn(
                tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            )


def flatten(x):
    shape = x.get_shape().as_list()
    dim = 1
    for d in shape[1:]:
        dim *= d
    x = tf.reshape(x, [-1, dim])
    return x

class Progbar(object):
    """
    Progbar class copied from keras (https://github.com/fchollet/keras/)
    Displays a progress bar.
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=[], exact=[]):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far+n, values)


def to_onehot(labels):
    """
    Args:
        labels: labels of shape(n_train)
    Return:
        one_hot: one_hot vector of shape (n_train, n_class)
    """
    n_class = 10
    one_hot = np.zeros((labels.size, n_class))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


def mini_batches(data, batch_size):
    """
        Args:
            data: (input, v, labels)
            batch_size: batch_size for SGD
    """

    (x, v, y) = data
    y = to_onehot(labels=y)
    indices = np.arange(v.size)
    np.random.shuffle(indices)

    all_batches = []

    for i in range(0, v.size / batch_size):
        x_batch = []
        v_batch = []
        y_batch = []

        start = i * batch_size

        for j in range(start, min(v.size, start + batch_size)):
            x_batch.append(x[indices[j]])
            v_batch.append(v[indices[j]])
            y_batch.append(y[indices[j]])

        all_batches.append((np.array(x_batch), np.array(v_batch), np.array(y_batch)))
    return all_batches
