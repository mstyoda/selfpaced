import tensorflow as tf

from model import Model
from tools import *
from config import *


class CnnModel(Model):

    def add_placeholders(self):
        self.input_placeholders = tf.placeholder(tf.float32, [None, Config.row_size, Config.column_size, Config.channel_size])
        self.label_placeholders = tf.placeholder(tf.float32, [None, Config.n_classes])
        self.dropout_placeholders = tf.placeholder(tf.float32, None)
        self.weight_placeholders = tf.placeholder(tf.float32, [None])

    def create_feed_dict(self, inputs_batch, weight_batch, labels_batch=None, dropout=1):
        feed_dict = {
            self.input_placeholders: inputs_batch,
            self.dropout_placeholders: dropout
        }
        if labels_batch is not None:
            feed_dict.update({self.label_placeholders: labels_batch})
        if weight_batch is not None:
            feed_dict.update({self.weight_placeholders: weight_batch})
        return feed_dict

    def add_prediction_op(self):
        """
            Add Cnn model here:
            input : ? * 32 * 32 * 3
            conv1 : 32 * 32 * 64
            maxpool : 16 * 16 * 64

            conv2 : 16 * 16 * 128
            maxpool : 8 * 8 * 128

            fc1 : 8192 * 4096
            fc2 : 4096 * 10
        """
        x = self.input_placeholders

        normal_initializer = tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32)

        conv1_filter = tf.get_variable(name='conv1_filter', shape=[3, 3, 3, 64], initializer=normal_initializer)
        conv1_bias = tf.get_variable(name='conv1_bias', shape=[64], initializer=normal_initializer)

        conv2_filter = tf.get_variable(name='conv2_filter', shape=[3, 3, 64, 128], initializer=normal_initializer)
        conv2_bias = tf.get_variable(name='conv2_bias', shape=[128], initializer=normal_initializer)

        fc1 = tf.get_variable(name='fc1', shape=[8192, 4096], initializer=normal_initializer)
        fc1_bias = tf.get_variable(name='fc1_bias', shape=[4096], initializer=normal_initializer)

        fc2 = tf.get_variable(name='fc2', shape=[4096, 10], initializer=normal_initializer)
        fc2_bias = tf.get_variable(name='fc2_bias', shape=[10], initializer=normal_initializer)

        conv1 = conv2D(x=x, filter=conv1_filter, bias=conv1_bias)
        conv2 = conv2D(x=maxpool(conv1), filter=conv2_filter, bias=conv2_bias)
        fc1_layer = tf.nn.relu(tf.nn.bias_add(value=tf.matmul(flatten(maxpool(conv2)), fc1), bias=fc1_bias))
        pred = tf.nn.relu(tf.nn.bias_add(value=tf.matmul(fc1_layer, fc2), bias=fc2_bias))

        pred_with_softmax = tf.nn.softmax(pred)

        return pred, pred_with_softmax

    def add_loss_op(self, pred):
        """
            Args:
                pred : pred tensor of shape (N_batch,N_class)
            Return:
                loss: loss in self-paced:
                    loss = (1/batch_size) sum_{i} v_i L(y_i,pred_i)
                loss_vector: origin softmax-ce loss of shape (N_batch)
        """
        loss_vector = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.label_placeholders,
            logits=pred)
        #loss = tf.reduce_mean(loss_vector * self.weight_placeholders)
        loss = tf.reduce_mean(loss_vector)
        return loss, loss_vector

    def add_training_op(self, loss):

        train_op = tf.train.AdamOptimizer(Config.lr).minimize(loss)
        return train_op

    def __init__(self, config):
        self.config = config
        self.save_path = 'saveModel/cnn_model.weights'
        self.build()
