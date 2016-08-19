import tensorflow as tf
from imdb import load_imdb_from_raw_cnn
import numpy as np
import argparse, sys

# Import MINST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters

def parse_args():

    parser = argparse.ArgumentParser(description = 'yolo training script')

    parser.add_argument('--imdb', type = str, help = 'image dataset')
    parser.add_argument('--snapshot', dest = 'snapshot', type = str, help = 'load weight from snapshot', default = None)
    parser.add_argument('--mode', type = str, dest = 'mode', help = 'train or test')
    parser.add_argument('--i', type = str, dest = 'image', help = 'input test image')
    parser.add_argument('--weights', type = str, dest = 'weights', help = 'weights for testing')
    # test net


    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

training_iters = 100000
display_step = 10
batch = 55
# Network Parameters
n_input = 448 # MNIST data input (img shape: 28*28)
n_classes = 35 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input ** 2 * 3])
y = tf.placeholder(tf.float32, [None, n_classes])
#keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

def log(tensor, string):

    return tf.Print(tensor, [tensor], string)
# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')
def avgpool2d(x, k = 2):
    return tf.nn.avg_pool(x, ksize = [1, k, k, 1], strides = [1, k, k, 1], padding = 'SAME')


# Create model
def conv_net(x, weights, biases, n_class):

    x = tf.reshape(x, shape = [-1, 448, 448, 3])

    # Convolution Layer

    conv1 = conv2d(x, weights['conv1'], biases['conv1'], strides = 1)
    conv1 = maxpool2d(conv1, k = 2)
    conv2 = conv2d(conv1, weights['conv2'], biases['conv2'], strides = 1)
    conv2 = maxpool2d(conv2, k = 2)
    conv3 = conv2d(conv2, weights['conv3'], biases['conv3'], strides = 1)
    conv3 = maxpool2d(conv3, k = 2)
    conv4 = conv2d(conv3, weights['conv4'], biases['conv4'], strides = 1)
    conv4 = maxpool2d(conv4, k = 2)
    conv5 = conv2d(conv4, weights['conv5'], biases['conv5'], strides = 1)
    conv5 = maxpool2d(conv5, k = 2)
    conv6 = conv2d(conv5, weights['conv6'], biases['conv6'], strides = 1)
    conv6 = avgpool2d(conv6, k = 2)
    conv7 = conv2d(conv6, weights['conv7'], biases['conv7'], strides = 1)
    conv8 = conv2d(conv7, weights['conv8'], biases['conv8'], strides = 1)
    conv9 = conv2d(conv8, weights['conv9'], biases['conv9'], strides = 1)

    fc1 = tf.reshape(conv9, [-1, weights['fc1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['fc1']), biases['fc1'])
    fc1 = tf.nn.relu(fc1)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return out


n_classes = 35

weights = {

            'conv1': tf.Variable(tf.random_normal([3, 3, 3, 16]),      name = 'w_conv1'),

            'conv2': tf.Variable(tf.random_normal([3, 3, 16, 32]),     name = 'w_conv2'),

            'conv3': tf.Variable(tf.random_normal([3, 3, 32, 64]),     name = 'w_conv3'),
            'conv4': tf.Variable(tf.random_normal([3, 3, 64, 128]),    name = 'w_conv4'),
            'conv5': tf.Variable(tf.random_normal([3, 3, 128, 256]),   name = 'w_conv5'),
            'conv6': tf.Variable(tf.random_normal([3, 3, 256, 512]),   name = 'w_conv6'),

            'conv7': tf.Variable(tf.random_normal([3, 3, 512, 1024]),  name = 'w_conv7'),
            'conv8': tf.Variable(tf.random_normal([3, 3, 1024, 1024]), name = 'w_conv8'),
            'conv9': tf.Variable(tf.random_normal([3, 3, 1024, 1024]), name = 'w_conv9'),
            
            'fc1': tf.Variable(tf.random_normal([7 * 7 * 1024, 4096]), name = 'w_fc1'),
            'out': tf.Variable(tf.random_normal([4096, n_classes]), name = 'w_out')
            
}

biases = {

            'conv1': tf.Variable(tf.random_normal([16]),   name = 'b_conv1'),

            'conv2': tf.Variable(tf.random_normal([32]),   name = 'b_conv2'),

            'conv3': tf.Variable(tf.random_normal([64]),   name = 'b_conv3'),
            'conv4': tf.Variable(tf.random_normal([128]),  name = 'b_conv4'),
            'conv5': tf.Variable(tf.random_normal([256]),  name = 'b_conv5'),
            'conv6': tf.Variable(tf.random_normal([512]),  name = 'b_conv6'),

            'conv7': tf.Variable(tf.random_normal([1024]), name = 'b_conv7'),
            'conv8': tf.Variable(tf.random_normal([1024]), name = 'b_conv8'),
            'conv9': tf.Variable(tf.random_normal([1024]), name = 'b_conv9'),
            
            'fc1':tf.Variable(tf.random_normal([4096]), name = 'b_fc1'),
            'out':tf.Variable(tf.random_normal([35]), name = 'b_out')
            

}


def save(session, saver, prefix, step):

    n = '{}_{}.model'.format(prefix, step)

    p = saver.save(session, n)

    return p

learning_rate = tf.train.exponential_decay(
            0.01,                # Base learning rate.
            0,  # Current index into the dataset.
            10000,          # Decay step.
            0.85,                # Decay rate.
            staircase=True)

arg = parse_args()

# Construct model
pred = conv_net(x, weights, biases, 1)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

cls = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z', '0','1','2','3','4','5','6','7','8','9']
n_class = len(cls)

images, objects, filename = load_imdb_from_raw_cnn('char', cls, batch)

conv_weights = [weights['conv1'],
             weights['conv2'],
             weights['conv3'],
             weights['conv4'],
             weights['conv5'],
             weights['conv6'],
             weights['conv7'],
             weights['conv8'],
             weights['conv9'],
             biases['conv1'],
             biases['conv2'],
             biases['conv3'],
             biases['conv4'],
             biases['conv5'],
             biases['conv6'],
             biases['conv7'],
             biases['conv8'],
             biases['conv9'],
             ]
# Launch the graph

with tf.Session() as sess:

    sess.run(init)

    step = 1

    pretrained_weights = tf.train.Saver(conv_weights) # only convolution weights
    pretrained_snapshot = tf.train.Saver() # contains fully connected weights


    if arg.snapshot is not None:

      pretrained_snapshot.restore(sess, arg.snapshot)

      step = int(arg.snapshot.split('_')[-1].split('.')[0]) / batch

      print 'load snapshot : {}'.format(arg.snapshot)

    # Keep training until reach max iterations
    
    while step * batch < training_iters:

        batch_x = images.next_batch()
        batch_y = objects.next_batch()

        # Run optimization op (backprop)
        sess.run(optimizer, { x : batch_x, y : batch_y })

        learning_rate = tf.train.exponential_decay(
            0.01,                # Base learning rate.
            step * batch,  # Current index into the dataset.
            10000,          # Decay step.
            0.85,                # Decay rate.
            staircase=True)

        
        if step % display_step == 0:

            print " ---------------------------------------"
            # Calculate batch loss and accuracy
            loss, acc, rate = sess.run([cost, accuracy, learning_rate], { x : batch_x,
                                                     y : batch_y, })
            print "Iter " + str(step*batch) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)
            print 'Learning rate : ', rate

        step += 1


    

    print "Optimization Finished!"

    p_w = save(sess, pretrained_weights, 'plate_pretrained_conv_weight', step * batch)
    p_s = save(sess, pretrained_snapshot, 'plate_pretrained_snapshot', step * batch)

    print 'pretrained conv weights saved : {}'.format(p_w)
    print 'pretrained snapshot saved : {}'.format(p_s)