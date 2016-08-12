from yolo_utils import cell_locate, convert_to_one, convert_to_reality
from imdbs import load_imdb_from_raw
from utils import IoU
from skimage import io
import tensorflow as tf
import numpy as np
import argparse
import sys
import cv2


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

def conv_net(x, weights, biases, n_class, dropout):

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
    conv6 = maxpool2d(conv6, k = 2)
    conv7 = conv2d(conv6, weights['conv7'], biases['conv7'], strides = 1)
    conv8 = conv2d(conv7, weights['conv8'], biases['conv8'], strides = 1)
    conv9 = conv2d(conv8, weights['conv9'], biases['conv9'], strides = 1)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    #fc1 = tf.reshape(conv9, [-1, 256])
   
    fc1 = tf.reshape(conv9, [-1, weights['fc1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['fc1']), biases['fc1'])
    fc1 = tf.nn.relu(fc1)
   
    # fc1 = tf.nn.dropout(fc1, dropout)

    fc2 = tf.add(tf.matmul(fc1, weights['fc2']), biases['fc2'])
    fc2 = tf.reshape(fc2, [-1, 4096])
    fc2 = tf.nn.relu(fc2)
    
    fc3 = tf.add(tf.matmul(fc2, weights['fc3']), biases['fc3'])
    fc3 = tf.reshape(fc3, [-1, S * S, n_class + 5 * B])
    fc3 = tf.nn.relu(fc3)

    return fc3

def init(session, saver, pretrained):

    pass

def save(session, saver, weights, prefix, step):

    n = '{}_{}.model'.format(prefix, step)

    p = saver.save(session, n)

    return p


def Confidence(pred, y):
    
    confidence = None

    for b in xrange(B):

        pred_x = tf.reshape(tf.slice(pred, [0, 0, b * 5 + 0], [-1, -1, 1]), [-1, S * S ])
        pred_y = tf.reshape(tf.slice(pred, [0, 0, b * 5 + 1], [-1, -1, 1]), [-1, S * S ])
        pred_w = tf.reshape(tf.slice(pred, [0, 0, b * 5 + 2], [-1, -1, 1]), [-1, S * S ])
        pred_h = tf.reshape(tf.slice(pred, [0, 0, b * 5 + 3], [-1, -1, 1]), [-1, S * S ])
   
        pred_bbox = [pred_x, pred_y, pred_w, pred_h]

        pred_bbox = convert_to_reality(pred_bbox, n_width, n_height, S)

        temp = tf.reshape(IoU(pred_bbox, y[:,:4]), [-1, S * S, 1])

        if type(confidence) is not tf.python.framework.ops.Tensor:

            confidence = temp

        else:
     
            confidence = tf.concat(2, [confidence, temp])

    assert confidence.dtype == tf.float32

    return confidence, pred_bbox[0]

def Center(groundtruth, batch):

    """

    check which cell is the center of the object located


    """

    c = None

    for b in xrange(B):

        # grid cell index tensor shape : [ batch , cells , one of bboxes ]
        # find out which grid cell is the bbox center located in

        grid_cell_index = tf.cast(tf.reshape(cell_locate((n_width,n_height), groundtruth, S), [-1]), tf.int32)

        # generate index for terrible tensorflow slicing tensor

        index = tf.range(0, batch)

        # pack index with grid cell index 

        indices = tf.cast(tf.pack([index, grid_cell_index], axis = 1), tf.int64)


        # set the " center variable " to one ( which is boolean type ), in terms of grid cell index  
        temp = tf.SparseTensor(indices = indices, values = tf.ones(batch), shape = [batch , S * S])
        center_sparse = tf.sparse_tensor_to_dense(temp)
        # center_sparse = tf.reshape(center_sparse, [-1, S * S, 1])

        # convert sparse tensor to dense tensor, and set others to 0 , represent that they are no responsible to the object

        
        if type(c) is not tf.python.framework.ops.Tensor:

            c = tf.reshape(center_sparse, [-1, S * S, 1])
          
        else:
           
            center_sparse = tf.reshape(center_sparse, [-1, S * S, 1])

            c = tf.concat(2, [c, center_sparse])

    return c
def Center2(groundtruth, batch):

    """

    check which cell is the center of the object located


    """

    c = None

    for b in xrange(1):

        # grid cell index tensor shape : [ batch , cells , one of bboxes ]
        # find out which grid cell is the bbox center located in

        grid_cell_index = tf.cast(tf.reshape(cell_locate((448,448), groundtruth, S), [-1]), tf.int32)

        # generate index for terrible tensorflow slicing tensor

        index = tf.range(0, batch)

        # pack index with grid cell index 

        indices = tf.cast(tf.pack([index, grid_cell_index], axis = 1), tf.int64)


        # set the " center variable " to one ( which is boolean type ), in terms of grid cell index  
        temp = tf.SparseTensor(indices = indices, values = tf.ones(batch), shape = [batch , S * S])

    return temp, indices, grid_cell_index
def Responsible(center, confidence):

    """

    chose the best predictor

    choose which detector is reponsible to object base on which bbox has the maximum IoU value with ground truth bbox, by setting the varaible to True

    others to False 

    In function "Center" , there are some cells, that its has more that one bboxes responsible to the object , so we multiply the center with confidence 

    range of value of center = { 0 , 1 } 

    1 * IoU

    1 * IoU

    that we can get the maximum IoU and the best detector

    """

    iou = tf.mul(center, confidence)

    # find out maximum IoU

    maximum_IoU = tf.reshape(tf.reduce_max(iou, 2), [-1, S * S, 1])

    # Create a same shape of tensor as iou 

    temp = tf.concat(2, [maximum_IoU, maximum_IoU])

    for b in xrange(B - 2):

        temp = tf.concat(2, [temp, maximum_IoU])

    maximum_IoU = temp

    # return the bool type tensor 

    res = tf.logical_and(tf.greater(iou, 0),tf.greater_equal(iou, maximum_IoU))
    #res = tf.greater_equal(iou, maximum_IoU)
    return res, maximum_IoU, iou

def Appear(confidence, batch):

    return tf.greater(tf.reduce_sum(confidence,2),tf.zeros((batch, S * S)))

"""

training

"""
def train(learning_rate, iters, batch, cls, dataset, n_bbox = 2, n_cell = 7, n_width = 448, n_height = 448, display = 20, threshold = 0.5, snapshot = None):
    
    print '[*] loading configurence '

    n_class = len(cls)

    # load dataset

    images, objects, filename = load_imdb_from_raw(dataset, cls)

    images = np.array(images)

    x = tf.placeholder(tf.float32, [None, n_input * 3]) # feed_dict (unknown batch , features)
    y = tf.placeholder(tf.float32, [None, n_class + 4]) # feed_dict (unknown batch, prob for each classes)
  
    is_obj = None
    not_obj = None
    loss = None

    response_threshold = threshold
    display_step = display

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
        
        'fc1':tf.Variable(tf.random_normal([7 * 7 * 1024, 256])),
        'fc2':tf.Variable(tf.random_normal([256, 4096])),
        'fc3':tf.Variable(tf.random_normal([4096, 7 * 7 * (n_class + 5 * B)])),
 
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
        
        'fc1':tf.Variable(tf.random_normal([256])),
        'fc2':tf.Variable(tf.random_normal([4096])),
        'fc3':tf.Variable(tf.random_normal([S * S * (n_class + 5 * B)])),

    }
    
    lcoord = tf.constant(5, dtype = tf.float32)
    lnoobj = tf.constant(0.5, dtype = tf.float32)
  
    # forward propagate

    pred = conv_net(x, weights, biases, n_class, 1)

    confidence, ppp = Confidence(pred, y)
    center = Center(y, batch)
    center2, indicess, grid_index = Center2(y, batch)
    responsible, maximum_IoU, iou = Responsible(center, confidence)
    appear = tf.cast(Appear(confidence, batch), tf.float32)


    not_responsible = tf.cast(tf.logical_not(responsible), tf.float32)
    responsible = tf.cast(responsible, tf.float32)

    # create loss function 
    
    for b in xrange(B):

        pred_x = tf.slice(pred, [0, 0, b * 5 + 0], [-1, -1, 1])
        gt_x = tf.reshape(tf.slice(y, [0,0], [-1,1]), [-1, 1, 1])

        pred_y = tf.slice(pred, [0, 0, b * 5 + 1], [-1, -1, 1])
        gt_y = tf.reshape(tf.slice(y, [0,1], [-1,1]), [-1, 1, 1])

        pred_w = tf.slice(pred, [0, 0, b * 5 + 2], [-1, -1, 1])
        gt_w = tf.reshape(tf.slice(y, [0,2], [-1,1]), [-1, 1, 1])

        pred_h = tf.slice(pred, [0, 0, b * 5 + 3], [-1, -1, 1])
        gt_h = tf.reshape(tf.slice(y, [0,3], [-1,1]), [-1, 1, 1])

        pred_c = tf.slice(pred, [0, 0, b * 5 + 4], [-1, -1, 1])
        gt_c = 1

        bbox = [gt_x, gt_y, gt_w, gt_h]

        # convert the x , y to the offset of grid cells and w , h to the range of 0 to 1 by divided by image size

        # offset x , offset y will only located in one grid cells , others will be set to 0 by the variable " Responsible "

        gt_x, gt_y, gt_w, gt_h = convert_to_one(bbox, n_width, n_height, S)

        dx = tf.pow(tf.sub(tf.cast(pred_x, tf.float32), tf.cast(gt_x, tf.float32)), 2)
        dy = tf.pow(tf.sub(tf.cast(pred_y, tf.float32), tf.cast(gt_y,tf.float32)), 2)
        dw = tf.pow(tf.sub(tf.cast(tf.pow(pred_w,0.5), tf.float32), tf.cast(tf.pow(gt_w,0.5), tf.float32)), 2)
        dh = tf.pow(tf.sub(tf.cast(tf.pow(pred_h,0.5), tf.float32), tf.cast(tf.pow(gt_h,0.5), tf.float32)), 2)
        dc = tf.pow(tf.sub(pred_c, gt_c), 2)

        loss_coord_xy = tf.mul(tf.mul(lcoord, tf.slice(responsible,[0,0,b],[-1,-1,1])), tf.add(dx,dy))     
        loss_coord_wh = tf.mul(tf.mul(lcoord, tf.slice(responsible,[0,0,b],[-1,-1,1])), tf.add(dw,dh))
        loss_is_obj = tf.mul(tf.slice(responsible,[0,0,b],[-1,-1,1]),dc)
        loss_no_obj = tf.mul(tf.mul(tf.slice(not_responsible,[0,0,b],[-1,-1,1]),dc), lnoobj)

        if loss == None:

            # print 'is None loss : ', loss_coord_wh.get_shape()

            loss = tf.add(tf.add(loss_coord_xy,loss_coord_wh), tf.add(loss_is_obj,loss_no_obj))

        else:

            # print 'is loss : ' ,loss_coord_wh.get_shape()

            loss = tf.add(loss, tf.add(tf.add(loss_coord_xy,loss_coord_wh), tf.add(loss_is_obj,loss_no_obj)))

            # print 'is loss is loss : ' ,loss.get_shape()

        index = b + 1

    """
    reshape loss [batch, cell, bbox] to [batch, bbox], so we can sum over all bbox

    """

    loss = tf.reshape(tf.reduce_sum(loss,1), [-1])

    y_cls = tf.reshape(tf.slice(pred, [0,0,5 * index], [-1,-1,-1]),[-1, S * S, n_class])

    pred_cls = tf.slice(pred, [0,0,5 * index], [-1,-1,-1])

    gt_cls = tf.pow(tf.sub(y_cls, pred_cls), 2)

    is_appear = tf.reshape(appear, [-1, S * S, 1])
    gt_cls = tf.mul(is_appear, gt_cls)
    gt_cls = tf.reduce_sum(gt_cls,1)
    gt_cls = tf.reduce_sum(gt_cls,1)
 
    loss = tf.add(loss, gt_cls)

    assert int(tf.slice(y,[0,4],[-1,-1]).get_shape()[1]) == n_class

    loss = tf.reduce_mean(loss)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:

        saver = tf.train.Saver()

        if snapshot is not None:

            saver.restore(sess, snapshot)

            print '[*] load from snapshot : {}'.format(snapshot)

            step = int(snapshot.split('_')[-1].split('.')[0]) / batch

            # some learning rate modify 

        else:

            init = tf.initialize_all_variables()

            sess.run(init)

            step = 0

        print '[*] start training ... '

        while step * batch < training_iters:

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
            
            assert batch_y.shape[-1] == int(y.get_shape()[-1]) and len(batch_x) == batch

            sess.run(optimizer, feed_dict = {
                                    x:batch_x,
                                    y:batch_y})

            if step % display_step == 0:

                cost = sess.run([loss],feed_dict = {
                                        x:batch_x,
                                        y:batch_y})

                print "Iter " , str(step * batch) + ", Minibatch Loss = " , cost
            
            step += 1

        p = save(sess, saver, weights, 'plate', step * batch)

        print '[*] Model saved in file : {}'.format(p)

        print "[*] Optimization Finished!"




if __name__ == '__main__':

    args = parse_args()

    batch = 64
    display = 1
    dataset = '5000_raw'
        #dataset = 'char'
    n_width = 448
    n_height = 448
    n_input = n_width * n_height
    cls = ['plate']
        #cls = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z', '0','1','2','3','4','5','6','7','8','9']

        #assert len(cls) == 35

    B = 2
    S = 7

    if args.mode == 'train':

        
        learning_rate = 0.00001
        training_iters = 30000
        train(learning_rate, training_iters, batch, cls, dataset,display = 1, snapshot = args.snapshot)

    elif args.mode == 'test':

        x = tf.placeholder(tf.float32, [n_input * 3])
        image = io.imread(args.image, False)
        image = cv2.resize(image, (448, 448))

        image = image.flatten()
        n_class = len(cls)
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
            
            'fc1':tf.Variable(tf.random_normal([7 * 7 * 1024, 256])),
            'fc2':tf.Variable(tf.random_normal([256, 4096])),
            'fc3':tf.Variable(tf.random_normal([4096, 7 * 7 * (n_class + 5 * B)])),
     
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
            
            'fc1':tf.Variable(tf.random_normal([256])),
            'fc2':tf.Variable(tf.random_normal([4096])),
            'fc3':tf.Variable(tf.random_normal([S * S * (n_class + 5 * B)])),

        }



        pred = conv_net(x, weights, biases, len(cls), 1)



        with tf.Session() as sess:

            saver = tf.train.Saver()

            if args.weights is not None:

                saver.restore(sess, args.weights)

                print '[*] load weights from : {}'.format(args.weights)

                # some learning rate modify 

            else:

                print 'required weights for testing ... '

               
            p = sess.run(pred, feed_dict = {x : image})

            print 'p shape : ', p.shape


            for b in xrange(B):

                pred_x = p[:,:,b*5+0]
                pred_y = p[:,:,b*5+1]
                pred_w = p[:,:,b*5+2]
                pred_h = p[:,:,b*5+3]
               
                pred_bbox = [pred_x, pred_y, pred_w, pred_h]

                pred_bbox = convert_to_reality(pred_bbox, n_width, n_height, S)

               # print 'box : [{}]'.format(sess.run(pred_bbox))
                print 'x : {}'.format(sess.run(pred_bbox[0]))
                print 'y : {}'.format(sess.run(pred_bbox[1]))
                print 'w : {}'.format(sess.run(pred_bbox[2]))
                print 'h : {}'.format(sess.run(pred_bbox[3]))
                print '--------------------------------------------------'