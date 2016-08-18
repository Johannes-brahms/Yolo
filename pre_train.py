from yolo_utils import cell_locate, convert_to_one, convert_to_reality
from imdbs import load_imdb_from_raw_cnn
from utils import IoU
from skimage import io
import tensorflow as tf
import numpy as np
import argparse
import sys
import cv2

def conv2d(x, W, b, strides = 1):

    # input [ batch , height , width , channels ]
    # filters [ width , height , channels , output channels (number of filters)]
    #print x.get_shape()
    #print W.get_shape()
    x = tf.nn.conv2d(x, W, strides = [1, strides, strides, 1], padding = 'SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k = 2):
    return tf.nn.max_pool(x, ksize = [1, k, k, 1], strides = [1, k, k, 1], padding = 'SAME')


def avgpool2d(x, k = 2):
    return tf.nn.avg_pool(x, ksize = [1, k, k, 1], strides = [1, k, k, 1], padding = 'SAME')



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

    return fc1


n_classes = 35

weights = {

            'conv1': tf.Variable(tf.random_uniform([3, 3, 3, 16], minval = -0.5, maxval = 0.5)),

            'conv2': tf.Variable(tf.random_uniform([3, 3, 16, 32], minval = -0.5, maxval = 0.5)),

            'conv3': tf.Variable(tf.random_uniform([3, 3, 32, 64], minval = -0.5, maxval = 0.5)),
            'conv4': tf.Variable(tf.random_uniform([3, 3, 64, 128], minval = -0.5, maxval = 0.5)),
            'conv5': tf.Variable(tf.random_uniform([3, 3, 128, 256], minval = -0.5, maxval = 0.5)),
            'conv6': tf.Variable(tf.random_uniform([3, 3, 256, 512], minval = -0.5, maxval = 0.5)),

            'conv7': tf.Variable(tf.random_uniform([3, 3, 512, 1024], minval = -0.5, maxval = 0.5)),
            'conv8': tf.Variable(tf.random_uniform([3, 3, 1024, 1024], minval = -0.5, maxval = 0.5)),
            'conv9': tf.Variable(tf.random_uniform([3, 3, 1024, 1024], minval = -0.5, maxval = 0.5)),
            
            'fc1':tf.Variable(tf.random_normal([7 * 7 * 1024, n_classes])),
            
}

biases = {

            'conv1': tf.Variable(tf.random_normal([16])),

            'conv2': tf.Variable(tf.random_normal([32])),

            'conv3': tf.Variable(tf.random_normal([64])),
            'conv4': tf.Variable(tf.random_normal([128])),
            'conv5': tf.Variable(tf.random_normal([256])),
            'conv6': tf.Variable(tf.random_normal([512])),

            'conv7': tf.Variable(tf.random_normal([1024])),
            'conv8': tf.Variable(tf.random_normal([1024])),
            'conv9': tf.Variable(tf.random_normal([1024])),
            
            'fc1':tf.Variable(tf.random_normal([35])),
            

}



# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units
cls = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z', '0','1','2','3','4','5','6','7','8','9']
n_class = len(cls)

images, objects, filename = load_imdb_from_raw_cnn('char', cls)

images = np.array(images)

x = tf.placeholder(tf.float32, [None, 448 * 448 * 3]) # feed_dict (unknown batch , features)
y = tf.placeholder(tf.float32, [None, n_class]) # feed_dict (unknown batch, prob for each classes)
  
pred = conv_net(x, weights, biases, 1)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))


optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

            
batch = 5

with tf.Session() as sess:

    sess.run(init)

    step = 0

      

    while step < training_iters:

        batch_x = []
        batch_y = []
        start = step * batch % len(images)
        end = (step + 1) * batch % len(images) # from zero

        if end < start:
            batch_x = np.vstack((images[start : ],images[:end]))
            batch_y = np.vstack((objects[start : ],objects[:end]))
                #_filename = np.vstack((filename[start :], filename[:end]))
        else:
            batch_x = images[start : end]
            batch_y = objects[start : end]
                #_filename = filename[start : end]
            
        print " ---------------------------------------"


        sess.run(optimizer, feed_dict = {
                                    x:batch_x,
                                    y:batch_y})

        print batch_x[0]
        print batch_y[0]

        if step % display_step == 0:

                loss,  = sess.run([cost],feed_dict = {
                                        x:batch_x,
                                        y:batch_y})



        print "Iter " , str(step) + " , Minibatch Loss = " , loss #" , Learning rate : ", learning_rate

       
        step += 1

        if step % 100 == 0:

            learning_rate *= 0.8 ** (step / 100)

    p = save(sess, saver, weights, 'plate', step * batch)

    print '[*] Model saved in file : {}'.format(p)

    print "[*] Optimization Finished!"



