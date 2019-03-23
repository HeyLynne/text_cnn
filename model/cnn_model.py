#coding=utf-8
import tensorflow as tf

class TextCNNConfig(object):
    """
    Text CNN config
    """
    embedding_dim = 64 # word enbeding dimension
    seq_length = 600
    num_classes = 10
    num_filters = 256 # kernel numbers
    kerner_size = 5
    vocab_size = 5000

    hidden_dim = 128

    dropout_keep_prob = 0.5
    learning_rate = 1e-3

    batch_size = 64
    num_epochs = 10

    print_per_batch = 100
    save_per_batch = 10

class TextCNN(object):
    def __init__(self, config):
        self.config = config
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name = 'input_x')
        self.input_y = tf.placeholder(tf.int32, [None, self.config.num_classes], name = 'input_y')
        self.keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
        self.cnn()

    def cnn(self):
        """
        CNN model
        """
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
        with tf.name_scope("cnn"):
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kerner_size, name = 'conv')
            gmp = tf.reduce_max(conv, reduction_indices = [1], name = 'gmp')

        with tf.name_scope('score'):
            fc1 = tf.layers.dense(gmp, self.config.hidden_dim, name = 'fc1')
            fc1 = tf.contrib.layers.dropout(fc1, self.keep_prob)
            fc1 = tf.nn.relu(fc1)

            self.logits = tf.layers.dense(fc1, self.config.num_classes, name = 'fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1) # predict sentence classification

        with tf.name_scope('optimize'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.input_y)
            self.loss = tf.reduce_min(cross_entropy)
            self.optim = tf.train.AdamOptimizer(learning_rate = self.config.learning_rate).minimize(self.loss)

        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
