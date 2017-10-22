import tensorflow as tf

from model import Model
from tools import *
from config import *


class VggModel(Model):

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
            Add VGG(19 layers) model here:
            - input :  ? * 32 * 32 * 3
            - conv1_1 : 32 * 32 * 64
            - conv1_2 : 32 * 32 * 64
            - maxpool

            - conv2_1 : 16 * 16 * 128
            - conv2_2 : 16 * 16 * 128
            - maxpool

            - conv3_1 : 8 * 8 * 256
            - conv3_2 : 8 * 8 * 256
            - conv3_3 : 8 * 8 * 256
            - conv3_4 : 8 * 8 * 256
            - maxpool

            - conv4_1 : 4 * 4 * 512
            - conv4_2 : 4 * 4 * 512
            - conv4_3 : 4 * 4 * 512
            - conv4_4 : 4 * 4 * 512
            - maxpool

            - conv5_1 : 2 * 2 * 512
            - conv5_2 : 2 * 2 * 512
            - conv5_3 : 2 * 2 * 512
            - conv5_4 : 2 * 2 * 512
            - maxpool

            - fc6 : 512 * 4096
            - fc7 : 4096 * 4096
            - fc8 : 4096 * 4096
        """
        x = tf.nn.lrn(self.input_placeholders)

        normal_initializer = normal_init()

        conv1_1 = normal_initializer('conv1_1', [3, 3, 3, 64])
        conv1_1_bias = normal_initializer('cov1_1_bias', [64])

        conv1_2 = normal_initializer('conv1_2', [3, 3, 64, 64])
        conv1_2_bias = normal_initializer('conv1_2_bias', [64])

        conv2_1 = normal_initializer('conv2_1', [3, 3, 64, 128])
        conv2_1_bias = normal_initializer('conv2_1_bias', [128])
        conv2_2 = normal_initializer('conv2_2', [3, 3, 128, 128])
        conv2_2_bias = normal_initializer('conv2_2_bias', [128])

        conv3_1 = normal_initializer('conv3_1', [3, 3, 128, 256])
        conv3_1_bias = normal_initializer('conv3_1_bias', [256])
        conv3_2 = normal_initializer('conv3_2', [3, 3, 256, 256])
        conv3_2_bias = normal_initializer('conv3_2_bias', [256])
        conv3_3 = normal_initializer('conv3_3', [3, 3, 256, 256])
        conv3_3_bias = normal_initializer('conv3_3_bias', [256])
        conv3_4 = normal_initializer('conv3_4', [3, 3, 256, 256])
        conv3_4_bias = normal_initializer('conv3_4_bias', [256])

        conv4_1 = normal_initializer('conv4_1', [3, 3, 256, 512])
        conv4_1_bias = normal_initializer('conv4_1_bias', [512])
        conv4_2 = normal_initializer('conv4_2', [3, 3, 512, 512])
        conv4_2_bias = normal_initializer('conv4_2_bias', [512])
        conv4_3 = normal_initializer('conv4_3', [3, 3, 512, 512])
        conv4_3_bias = normal_initializer('conv4_3_bias', [512])
        conv4_4 = normal_initializer('conv4_4', [3, 3, 512, 512])
        conv4_4_bias = normal_initializer('conv4_4_bias', [512])

        conv5_1 = normal_initializer('conv5_1', [3, 3, 512, 512])
        conv5_1_bias = normal_initializer('conv5_1_bias', [512])
        conv5_2 = normal_initializer('conv5_2', [3, 3, 512, 512])
        conv5_2_bias = normal_initializer('conv5_2_bias', [512])
        conv5_3 = normal_initializer('conv5_3', [3, 3, 512, 512])
        conv5_3_bias = normal_initializer('conv5_3_bias', [512])
        conv5_4 = normal_initializer('conv5_4', [3, 3, 512, 512])
        conv5_4_bias = normal_initializer('conv5_4_bias', [512])

        fc6 = normal_initializer('fc6', [512, 4096])
        fc6_bias = normal_initializer('fc6_bias', [4096])
        fc7 = normal_initializer('fc7', [4096, 4096])
        fc7_bias = normal_initializer('fc7_bias', [4096])
        fc8 = normal_initializer('fc8', [4096, Config.n_classes])
        fc8_bias = normal_initializer('fc8_bias', [Config.n_classes])

        layer1 = conv2D(x, conv1_1, conv1_1_bias)
        layer1 = conv2D(layer1, conv1_2, conv1_2_bias)
        layer1 = maxpool(layer1)

        layer2 = conv2D(layer1, conv2_1, conv2_1_bias)
        layer2 = conv2D(layer2, conv2_2, conv2_2_bias)
        layer2 = maxpool(layer2)

        layer3 = conv2D(layer2, conv3_1, conv3_1_bias)
        layer3 = conv2D(layer3, conv3_2, conv3_2_bias)
        layer3 = conv2D(layer3, conv3_3, conv3_3_bias)
        layer3 = conv2D(layer3, conv3_4, conv3_4_bias)
        layer3 = maxpool(layer3)

        layer4 = conv2D(layer3, conv4_1, conv4_1_bias)
        layer4 = conv2D(layer4, conv4_2, conv4_2_bias)
        layer4 = conv2D(layer4, conv4_3, conv4_3_bias)
        layer4 = conv2D(layer4, conv4_4, conv4_4_bias)
        layer4 = maxpool(layer4)

        layer5 = conv2D(layer4, conv5_1, conv5_1_bias)
        layer5 = conv2D(layer5, conv5_2, conv5_2_bias)
        layer5 = conv2D(layer5, conv5_3, conv5_3_bias)
        layer5 = conv2D(layer5, conv5_4, conv5_4_bias)
        layer5 = maxpool(layer5)

        layer6 = flatten(layer5)

        layer6 = tf.nn.relu(tf.nn.bias_add(tf.matmul(layer6, fc6), fc6_bias))
        layer7 = tf.nn.relu(tf.nn.bias_add(tf.matmul(layer6, fc7), fc7_bias))

        pred = tf.nn.bias_add(tf.matmul(layer7, fc8), fc8_bias)
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
        # loss = tf.reduce_mean(loss_vector * self.weight_placeholders)
        loss = tf.reduce_mean(loss_vector)
        return loss, loss_vector

    def add_training_op(self, loss):
        train_op = tf.train.AdadeltaOptimizer(Config.lr).minimize(loss)
        return train_op

    def __init__(self, config):
        self.config = config
        self.save_path = 'saveModel/vgg_model.weights'
        self.build()
