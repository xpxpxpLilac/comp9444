"""
All tensorflow objects, if not otherwise specified, should be explicity
created with tf.float32 datatypes. Not specifying this datatype for variables and
placeholders will cause your code to fail some tests.

You do not need to import any other libraries for this assignment.

Along with the provided functional prototypes, there is another file,
"train.py" which calls the functions listed in this file. It trains the
specified network on the MNIST dataset, and then optimizes the loss using a
standard gradient decent optimizer. You can run this code to check the models
you create in part II.
"""

import tensorflow as tf
import pdb
""" PART I """


def add_consts():
    """
    EXAMPLE:
    Construct a TensorFlow graph that declares 3 constants, 5.1, 1.0 and 5.9
    and adds these together, returning the resulting tensor.
    """
    c1 = tf.constant(5.1)
    c2 = tf.constant(1.0)
    c3 = tf.constant(5.9)
    a1 = tf.add(c1, c2)
    af = tf.add(a1, c3)
    return af


def add_consts_with_placeholder():
    """ 
    Construct a TensorFlow graph that constructs 2 constants, 5.1, 1.0 and one
    TensorFlow placeholder of type tf.float32 that accepts a scalar input,
    and adds these three values together, returning as a tuple, and in the
    following order:
    (the resulting tensor, the constructed placeholder).
    """
    c1 = tf.constant(5.1)
    c2 = tf.constant(1.0)
    c3 = tf.placeholder(dtype= tf.float32)
    a1 = tf.add(c1, c2)
    af = tf.add(a1, c3)

    return af, c3


def my_relu(in_value):
    """
    Implement a ReLU activation function that takes a scalar tf.placeholder as input
    and returns the appropriate output. For more information see the assignment spec.
    """
    zero = tf.constant(0.)
    out_value = tf.cond(in_value < zero, lambda: 0., lambda: in_value)
  
    return out_value


def my_perceptron(x):
    """
    Implement a single perception that takes four inputs and produces one output,
    using the RelU activation function you defined previously.

    Specifically, implement a function that takes a list of 4 floats x, and
    creates a tf.placeholder the same length as x. Then create a trainable TF
    variable that for the weights w. Ensure this variable is
    set to be initialized as all ones.

    Multiply and sum the weights and inputs following the peceptron outlined in the
    lecture slides. Finally, call your relu activation function.
    hint: look at tf.get_variable() and the initalizer argument.
    return the placeholder and output in that order as a tuple

    Note: The code will be tested using the following init scheme
        # graph def (your code called)
        init = tf.global_variables_initializer()

        self.sess.run(init)
        # tests here

    """
    i = tf.placeholder(tf.float32,shape=[x])
    weig = tf.get_variable(name='weig',shape=[x],initializer=tf.ones_initializer())
    dot = tf.tensordot(i, weig, 1)

    out = my_relu(dot)

    return i, out


""" PART II """
fc_count = 0  # count of fully connected layers. Do not remove.


def input_placeholder():
    return tf.placeholder(dtype=tf.float32, shape=[None, 784],
                          name="image_input")


def target_placeholder():
    return tf.placeholder(dtype=tf.float32, shape=[None, 10],
                          name="image_target_onehot")




def onelayer(X, Y, layersize=10):
    """
    Create a Tensorflow model for logistic regression (i.e. single layer NN)

    :param X: The input placeholder for images from the MNIST dataset
    :param Y: The output placeholder for image labels
    :return: The following variables should be returned  (variables in the
    python sense, not in the Tensorflow sense, although some may be
    Tensorflow variables). They must be returned in the following order.
        w: Connection weights
        b: Biases
        logits: The input to the activation function
        preds: The output of the activation function (a probability
        distribution over the 10 digits)
        batch_xentropy: The cross-entropy loss for each image in the batch
        batch_loss: The average cross-entropy loss of the batch
    """
    """
    w --tf.Variable is a matrix of [784,10] start with ones
    b --tf.Variable is matrix of [none, 10] start with ones
    use softmax as activation
    use cross-entropy for loss analyst
    start training - descent-gradient
    """
    #learn_rate = 1
    # with tf.variable_scope("one_layer",reuse=True) as layer1:
    #w = tf.get_variable(name='weight_for_one',shape=[784,10], initializer=tf.ones_initializer,trainable=False)
    w = tf.Variable(tf.zeros([784, layersize]), name='w')
    #b = tf.get_variable(name='b',shape=[1,10],initializer=tf.ones_initializer)
    b = tf.Variable(tf.zeros([1, layersize]), name='b')

    y_ = tf.matmul(X,w)
    logits = tf.add(y_,b)
    preds = tf.nn.softmax(logits)
    batch_xentropy = -Y * tf.log(preds)
    batch_loss = tf.reduce_mean(tf.reduce_sum(batch_xentropy, 1))

    #train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(batch_loss)

    return w, b, logits, preds, batch_xentropy, batch_loss


def twolayer(X, Y, hiddensize=30, outputsize=10):
    """
    Create a Tensorflow model for a Neural Network with one hidden layer

    :param X: The  input placeholder for images from the MNIST dataset
    :param Y: The output placeholder for image labels
    :return: The following variables should be returned in the following order.
        W1: Connection weights for the first layer
        b1: Biases for the first layer
        W2: Connection weights for the second layer
        b2: Biases for the second layer
        logits: The inputs to the activation function
        preds: The outputs of the activation function (a probability
        distribution over the 10 digits)
        batch_xentropy: The cross-entropy loss for each image in the batch
        batch_loss: The average cross-entropy loss of the batch
    """
    #layer one
    w1 = tf.Variable(tf.zeros([784, hiddensize]), name='w1')
    b1 = tf.Variable(tf.ones([1,hiddensize]), name='b1')
    #layer two
    w2 = tf.Variable(tf.random_normal([hiddensize, outputsize]), name='w2')
    b2 = tf.Variable(tf.random_normal([1, outputsize]), name='b2')
    mid_1 = tf.matmul(X,w1)
    hidden_sum = tf.add(mid_1,b1)
    hidden_input = tf.nn.relu(hidden_sum)
    
    mid_2 = tf.matmul(hidden_input,w2)
    logits = tf.add(mid_2,b2)
    preds = tf.nn.softmax(logits)

    batch_xentropy = -Y * tf.log(preds)
    batch_loss = tf.reduce_mean(tf.reduce_sum(batch_xentropy, 1))

    return w1, b1, w2, b2, logits, preds, batch_xentropy, batch_loss


def convnet(X, Y, convlayer_sizes=[10, 10], \
            filter_shape=[3, 3], outputsize=10, padding="same"):
    """
    Create a Tensorflow model for a Convolutional Neural Network. The network
    should be of the following structure:
    conv_layer1 -> conv_layer2 -> fully-connected -> output

    :param X: The  input placeholder for images from the MNIST dataset
    :param Y: The output placeholder for image labels
    :return: The following variables should be returned in the following order.
        conv1: A convolutional layer of convlayer_sizes[0] filters of shape filter_shape
        conv2: A convolutional layer of convlayer_sizes[1] filters of shape filter_shape
        w: Connection weights for final layer
        b: biases for final layer
        logits: The inputs to the activation function
        preds: The outputs of the activation function (a probability
        distribution over the 10 digits)
        batch_xentropy: The cross-entropy loss for each image in the batch
        batch_loss: The average cross-entropy loss of the batch

    hints:
    1) consider tf.layer.conv2d
    2) the final layer is very similar to the onelayer network. Only the input
    will be from the conv2 layer. If you reshape the conv2 output using tf.reshape,
    you should be able to call onelayer() to get the final layer of your network
    """

    conv1 = tf.layers.conv2d(
        X,
        convlayer_sizes[0],
        filter_shape,
        padding=padding,
        activation=tf.nn.relu,
        use_bias=True,
        kernel_initializer=tf.zeros_initializer(),
        bias_initializer=tf.zeros_initializer(),
        #trainable=True,
        # name='conv1',
        # reuse=True
    )

    conv2 = tf.layers.conv2d(
        conv1,
        convlayer_sizes[1],
        filter_shape,
        padding=padding,
        activation=tf.nn.relu,
        use_bias=True,
        kernel_initializer=tf.ones_initializer(),
        bias_initializer=tf.zeros_initializer(),
        #trainable=True,
        # name='conv2',
        # reuse=True
    )

    out = tf.reshape(conv2,[-1,784*convlayer_sizes[1]])

    w = tf.Variable(tf.zeros([out.get_shape()[1], outputsize]), name='w')
    b = tf.Variable(tf.zeros([1, outputsize]), name='b')

    y_ = tf.matmul(out,w)
    logits = tf.add(y_,b)
    preds = tf.nn.softmax(logits)
    batch_xentropy = -Y * tf.log(preds)
    batch_loss = tf.reduce_mean(tf.reduce_sum(batch_xentropy, 1))

    return conv1, conv2, w, b, logits, preds, batch_xentropy, batch_loss


def train_step(sess, batch, X, Y, train_op, loss_op, summaries_op):
    """
    Run one step of training.

    :param sess: the current session
    :param batch: holds the inputs and target outputs for the current minibatch
    batch[0] - array of shape [minibatch_size, 784] with each row holding the
    input images
    batch[1] - array of shape [minibatch_size, 10] with each row holding the
    one-hot encoded targets
    :param X: the input placeholder
    :param Y: the output target placeholder
    :param train_op: the tensorflow operation that will run one step of training
    :param loss_op: the tensorflow operation that will return the loss of your
    model on the batch input/output

    :return: a 3-tuple: train_op_result, loss, summary
    which are the results of running the train_op, loss_op and summaries_op
    respectively.
    """
    train_result, loss, summary = \
        sess.run([train_op, loss_op, summaries_op], feed_dict={X: batch[0], Y: batch[1]})
    return train_result, loss, summary
