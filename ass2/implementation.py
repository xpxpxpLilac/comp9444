import tensorflow as tf
import re

BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 250  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector

LSTM_SIZE = 128
LSTM_LAYER = 2
LEARNING_RATE = 0.001

stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than'})

def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """
    review = str(review)
    # processed_review = review
    # change case
    review = review.lower()
    # remove the stop words
    word_list = review.split()
    resultwords  = [word for word in word_list if word not in stop_words]
    review = ' '.join(resultwords)
    # replace punctuations to blank space
    review = re.sub(r'[^\w\s]','',review)
    #strip
    processed_review = re.sub(r' +', ' ', review)
    processed_review = processed_review.split()
    return processed_review



def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    Note that the input is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """

    
    input_data = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE], name = 'input_data')
    labels = tf.placeholder(tf.int32, [BATCH_SIZE, 2], name = 'labels')
    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=(), name='dropout_keep_prob')
    lstm = tf.contrib.rnn.BasicLSTMCell(LSTM_SIZE)
    # cell = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=dropout_keep_prob)
    # initial_state = cell.zero_state(BATCH_SIZE, tf.float32)
    # outputs, _ = tf.nn.dynamic_rnn(cell, input_data, initial_state=initial_state)
    lstm_2 = tf.contrib.rnn.BasicLSTMCell(LSTM_SIZE)
    drop_out = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=dropout_keep_prob)
    rnn_cell = tf.contrib.rnn.MultiRNNCell([drop_out,lstm_2])
    initial_state = rnn_cell.zero_state(BATCH_SIZE, tf.float32)
    outputs, _ = tf.nn.dynamic_rnn(rnn_cell, input_data, initial_state=initial_state)


    w = tf.Variable(tf.random_normal([LSTM_SIZE, 2]), name='w')
    b = tf.Variable(tf.random_normal([2]), name='b')
    mid = tf.matmul(outputs[:,-1,:],w)
    logits = tf.add(mid,b)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels, logits = logits), name="loss")
    optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE).minimize(loss)

    preds_op = tf.nn.softmax(logits)
    correct_preds_op = tf.equal(tf.argmax(preds_op, 1), tf.argmax(labels, 1))
    Accuracy = tf.reduce_mean(tf.cast(correct_preds_op, tf.float32), name='accuracy')

    return input_data, labels, dropout_keep_prob, optimizer, Accuracy, loss
