from yolo_utils import cell_locate, convert_to_one
from imdbs import load_imdb_from_raw
from utils import IoU
from skimage import io
import tensorflow as tf
import numpy as np
import argparse
import sys


def parse_args():

    parser = argparse.ArgumentParser(description = 'yolo training script')

    parser.add_argument('--imdb', type = str, help = 'image dataset')
    parser.add_argument('--snapshot', dest = 'snapshot', type = str, help = 'load weight from snapshot', default = None)


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

    conv1 = conv2d(x, weights['conv1'], biases['conv1'], strides = 2)
    conv1 = maxpool2d(conv1, k = 2)

    conv2 = conv2d(conv1, weights['conv2'], biases['conv2'])
    conv2 = maxpool2d(conv2, k = 2)

    conv3 = conv2d(conv2, weights['conv3'], biases['conv3'])
    conv4 = conv2d(conv3, weights['conv4'], biases['conv4'])
    conv5 = conv2d(conv4, weights['conv5'], biases['conv5'])
    conv6 = conv2d(conv5, weights['conv6'], biases['conv6'])
    conv6 = maxpool2d(conv6, k = 2)

    """
    conv7 = conv2d(conv6, weights['conv7'], biases['conv7'])
    conv8 = conv2d(conv7, weights['conv8'], biases['conv8'])
    """

    conv9 = conv2d(conv6, weights['conv9'], biases['conv9'])
    conv10 = conv2d(conv9, weights['conv10'], biases['conv10'])
    """
    conv11 = conv2d(conv10, weights['conv11'], biases['conv11'])
    conv12 = conv2d(conv11, weights['conv12'], biases['conv12'])
    conv13 = conv2d(conv12, weights['conv13'], biases['conv13'])
    conv14 = conv2d(conv13, weights['conv14'], biases['conv14'])
    conv15 = conv2d(conv14, weights['conv15'], biases['conv15'])
    """
    conv16 = conv2d(conv10, weights['conv16'], biases['conv16'])
    conv16 = maxpool2d(conv16, k = 2)

    conv17 = conv2d(conv16, weights['conv17'], biases['conv17'])

    conv18 = conv2d(conv17, weights['conv18'], biases['conv18'])
    conv19 = conv2d(conv18, weights['conv19'], biases['conv19'])
    conv20 = conv2d(conv19, weights['conv20'], biases['conv20'])
    conv21 = conv2d(conv20, weights['conv21'], biases['conv21'])
    conv22 = conv2d(conv21, weights['conv22'], biases['conv22'], strides = 2)

    conv23 = conv2d(conv22, weights['conv23'], biases['conv23'])
    conv24 = conv2d(conv23, weights['conv24'], biases['conv24'])
    print 'conv1  : ', conv1.get_shape()
    print 'conv2  : ', conv2.get_shape()
    print 'conv3  : ', conv3.get_shape()
    print 'conv4  : ', conv4.get_shape()
    print 'conv5  : ', conv5.get_shape()
    print 'conv6  : ', conv6.get_shape()
    """
    print 'conv7  : ', conv7.get_shape()
    print 'conv8  : ', conv8.get_shape()
    """
    print 'conv9  : ', conv9.get_shape()
    print 'conv10  : ', conv10.get_shape()
    """
    print 'conv11  : ', conv11.get_shape()
    print 'conv12  : ', conv12.get_shape()
    print 'conv13  : ', conv13.get_shape()
    print 'conv14  : ', conv14.get_shape()
    print 'conv15  : ', conv15.get_shape()
    """
    print 'conv16  : ', conv16.get_shape()

    print 'conv17  : ', conv17.get_shape()
    print 'conv18  : ', conv18.get_shape()
    print 'conv19  : ', conv19.get_shape()
    print 'conv20  : ', conv20.get_shape()
    print 'conv21  : ', conv21.get_shape()
    print 'conv22  : ', conv22.get_shape()
    print 'conv23 : ', conv23.get_shape()
    print 'conv24 : ', conv24.get_shape()
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv24, [-1, weights['fc1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['fc1']), biases['fc1'])
    fc1 = tf.nn.relu(fc1)
    print 'fc1 : ' , fc1.get_shape()
    # fc1 = tf.nn.dropout(fc1, dropout)

    fc2 = tf.add(tf.matmul(fc1, weights['fc2']), biases['fc2'])
    fc2 = tf.reshape(fc2, [-1, S * S, n_class + 5 * B])
    fc2 = tf.nn.relu(fc2)

    #fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    #fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    #fc1 = tf.nn.relu(fc1)

    # Apply dropout
    # fc1 = tf.nn.dropout(fc1, dropout)

    #out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return fc2

def init(session, saver, pretrained):

    pass


def save(session, saver, weights, prefix, step):

    n = '{}_{}.model'.format(prefix, step)

    p = saver.save(session, n)

    return p


def Confidence(pred, y):
    
    confidence = None

    print 'prediction : ', pred.get_shape()
    print 'ground truth : ', y.get_shape()
    
    for b in xrange(B):

        shape = (-1, int(pred.get_shape()[1]), b + 1)    

        if type(confidence) is not tf.python.framework.ops.Tensor:

            confidence = IoU(tf.slice(pred,[0,0,b*5],[-1,-1,4]), y[:,:4])
        else:
            confidence = tf.concat(1,(confidence,IoU(tf.slice(pred,[0,0,b * 5],[-1,-1,4]), y[ : ,:4])))
            confidence = tf.reshape(confidence,shape)

    assert confidence.dtype == tf.float32

    print 'confidence shape ', confidence.get_shape()

    return confidence

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

        center_sparse = tf.SparseTensor(indices = indices, values = tf.ones(batch), shape = [batch , S * S])

        # convert sparse tensor to dense tensor, and set others to 0 , represent that they are no responsible to the object

        if type(c) is not tf.python.framework.ops.Tensor:

            c = tf.sparse_tensor_to_dense(center_sparse)
        
        else:

            c = tf.concat(1,[c, tf.sparse_tensor_to_dense(center_sparse)])


    c = tf.reshape(c,[-1, S * S, B])
        
    return c

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

    maximum_IoU = tf.reduce_max(iou, 2)


    # Create a same shape of tensor as iou 

    for b in xrange(B - 1):

        maximum_IoU = tf.concat(1, [maximum_IoU, maximum_IoU])

    maximum_IoU = tf.reshape(maximum_IoU, [-1, S * S, B])

    # return the bool type tensor 

    res = tf.greater_equal(center, maximum_IoU)

    return res

def Appear(confidence, batch):

    return tf.greater(tf.reduce_sum(confidence,2),tf.zeros((batch, S * S)))

def convert_to_reality():
    pass

"""

training

"""
def train(learning_rate, iters, batch, cls, n_bbox = 2, n_cell = 7, n_width = 448, n_height = 448, display = 20, threshold = 0.5, snapshot = None):
    
    print '[*] loading configurence '

    n_class = len(cls)

    x = tf.placeholder(tf.float32, [None, n_input * 3]) # feed_dict (unknown batch , features)
    y = tf.placeholder(tf.float32, [None, n_class + 4]) # feed_dict (unknown batch, prob for each classes)
  
    is_obj = None
    not_obj = None
    loss = None

    response_threshold = threshold
    display_step = display

    weights = {

        'conv1': tf.Variable(tf.random_normal([7, 7, 3, 64])),

        'conv2': tf.Variable(tf.random_normal([3, 3, 64, 192])),

        'conv3': tf.Variable(tf.random_normal([1, 1, 192, 128])),
        'conv4': tf.Variable(tf.random_normal([3, 3, 128, 256])),
        'conv5': tf.Variable(tf.random_normal([1, 1, 256, 256])),
        'conv6': tf.Variable(tf.random_normal([3, 3, 256, 512])),

        #'conv7': tf.Variable(tf.random_normal([1, 1, 512, 256])),
        #'conv8': tf.Variable(tf.random_normal([3, 3, 256, 512])),
        'conv9': tf.Variable(tf.random_normal([1, 1, 512, 256])),
        'conv10': tf.Variable(tf.random_normal([3, 3, 256, 512])),
        #'conv11': tf.Variable(tf.random_normal([1, 1, 512, 256])),
        #'conv12': tf.Variable(tf.random_normal([3, 3, 256, 512])),
        #'conv13': tf.Variable(tf.random_normal([1, 1, 512, 256])),
        #'conv14': tf.Variable(tf.random_normal([3, 3, 256, 512])),
        #'conv15': tf.Variable(tf.random_normal([1, 1, 512, 512])),
        'conv16': tf.Variable(tf.random_normal([3, 3, 512, 1024])),

        'conv17': tf.Variable(tf.random_normal([1, 1, 1024, 512])),
        'conv18': tf.Variable(tf.random_normal([3, 3, 512, 1024])),
        'conv19': tf.Variable(tf.random_normal([1, 1, 1024, 512])),
        'conv20': tf.Variable(tf.random_normal([3, 3, 512, 1024])),
        'conv21': tf.Variable(tf.random_normal([3, 3, 1024, 1024])),
        'conv22': tf.Variable(tf.random_normal([3, 3, 1024, 1024])),
        'conv23': tf.Variable(tf.random_normal([3, 3, 1024, 1024])),
        'conv24': tf.Variable(tf.random_normal([3, 3, 1024, 1024])),

        'fc1':tf.Variable(tf.random_normal([7 * 7 * 1024, 4096])),
        'fc2':tf.Variable(tf.random_normal([4096, 7 * 7 * (n_class + 5 * B)])),
 
    }

    biases = {

        'conv1': tf.Variable(tf.random_normal([64])),

        'conv2': tf.Variable(tf.random_normal([192])),

        'conv3': tf.Variable(tf.random_normal([128])),
        'conv4': tf.Variable(tf.random_normal([256])),
        'conv5': tf.Variable(tf.random_normal([256])),
        'conv6': tf.Variable(tf.random_normal([512])),

        #'conv7': tf.Variable(tf.random_normal([256])),
        #'conv8': tf.Variable(tf.random_normal([512])),
        'conv9': tf.Variable(tf.random_normal([256])),
        'conv10': tf.Variable(tf.random_normal([512])),
        #'conv11': tf.Variable(tf.random_normal([256])),
        #'conv12': tf.Variable(tf.random_normal([512])),
        #'conv13': tf.Variable(tf.random_normal([256])),
        #'conv14': tf.Variable(tf.random_normal([512])),
        #'conv15': tf.Variable(tf.random_normal([512])),
        'conv16': tf.Variable(tf.random_normal([1024])),

        'conv17': tf.Variable(tf.random_normal([512])),
        'conv18': tf.Variable(tf.random_normal([1024])),
        'conv19': tf.Variable(tf.random_normal([512])),
        'conv20': tf.Variable(tf.random_normal([1024])),
        'conv21': tf.Variable(tf.random_normal([1024])),
        'conv22': tf.Variable(tf.random_normal([1024])),
        'conv23': tf.Variable(tf.random_normal([1024])),
        'conv24': tf.Variable(tf.random_normal([1024])),

        'fc1':tf.Variable(tf.random_normal([4096])),
        'fc2':tf.Variable(tf.random_normal([S * S * (n_class + 5 * B)])),
        #'out':tf.Variable(tf.random_normal([])),
    }
    
    lcoord = tf.constant(5, dtype = tf.float32)
    lnoobj = tf.constant(0.5, dtype = tf.float32)
  
    # forward propagate

    pred = conv_net(x, weights, biases, n_class, 1)

    confidence = Confidence(pred, y)
    center = Center(y, batch)
    responsible = Responsible(center, confidence)
    appear = tf.cast(Appear(confidence, batch), tf.float32)
    not_responsible = tf.cast(tf.logical_not(responsible), tf.float32)
    responsible = tf.cast(responsible, tf.float32)

    # load dataset

    images, objects = load_imdb_from_raw(dataset, cls)
    images = np.array(images)

    # create loss function 
    
    for b in xrange(B):

        # print 'prediction : ', pred.get_shape()

        pred_x = tf.slice(pred, [0,0,b * 5 + 0], [-1,-1,1])
        gt_x = tf.reshape(tf.slice(y, [0,0], [-1,1]), [-1, 1, 1])

        pred_y = tf.slice(pred, [0,0,b * 5 + 1], [-1,-1,1])
        gt_y = tf.reshape(tf.slice(y, [0,1], [-1,1]), [-1, 1, 1])

        pred_w = tf.slice(pred, [0,0,b * 5 + 2], [-1,-1,1])
        gt_w = tf.reshape(tf.slice(y, [0,2], [-1,1]), [-1, 1, 1])

        pred_h = tf.slice(pred, [0,0,b * 5 + 3], [-1,-1,1])
        gt_h = tf.reshape(tf.slice(y, [0,3], [-1,1]), [-1, 1, 1])

        pred_c = tf.slice(pred, [0,0,b * 5 + 4], [-1,-1,1])
        gt_c = 1

        bbox = [gt_x, gt_y, gt_w, gt_h]

        # convert the x , y to the offset of grid cells and w , h to the range of 0 to 1 by divided by image size

        # offset x , offset y will only located in one grid cells , others will be set to 0 by the variable " Responsible "

        gt_x, gt_y, gt_w, gt_h = convert_to_one(bbox, n_width, n_height, S)

        dx = tf.pow(tf.sub(pred_x, gt_x), 2)
        dy = tf.pow(tf.sub(pred_y, gt_y), 2)
        dw = tf.pow(tf.sub(tf.pow(pred_w,0.5), tf.pow(gt_w,0.5)), 2)
        dh = tf.pow(tf.sub(tf.pow(pred_h,0.5), tf.pow(gt_h,0.5)), 2)
        dc = tf.pow(tf.sub(pred_c, gt_c), 2)

        if loss == None:

            loss_coord_xy = tf.mul(tf.mul(lcoord, tf.slice(responsible,[0,0,b],[-1,-1,1])), tf.add(dx,dy))
            loss_coord_wh = tf.mul(tf.mul(lcoord, tf.slice(responsible,[0,0,b],[-1,-1,1])), tf.add(dw,dh))
            loss_is_obj = tf.mul(tf.slice(responsible,[0,0,b],[-1,-1,1]),dc)
            loss_no_obj = tf.mul(tf.slice(not_responsible,[0,0,b],[-1,-1,1]),dc)

            # print 'is None loss : ', loss_coord_wh.get_shape()

            loss = tf.add(tf.add(loss_coord_xy,loss_coord_wh), tf.add(loss_is_obj,loss_no_obj))

        else:

            loss_coord_xy = tf.mul(tf.mul(lcoord, tf.slice(responsible,[0,0,b],[-1,-1,1])), tf.add(dx,dy))
            loss_coord_wh = tf.mul(tf.mul(lcoord, tf.slice(responsible,[0,0,b],[-1,-1,1])), tf.add(dw,dh))
            loss_is_obj = tf.mul(tf.slice(responsible,[0,0,b],[-1,-1,1]),dc)
            loss_no_obj = tf.mul(tf.slice(not_responsible,[0,0,b],[-1,-1,1]),dc)

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

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:

        

        saver = tf.train.Saver()

        if snapshot is not None:

            saver.restore(sess, snapshot)

            #tf.report_uninitialized_variables(var_list=[weights['conv1']], name='report_uninitialized_variables')

            ##rint tf.is_variable_initialized(weights['conv1'])

            print '[*] load from snapshot : {}'.format(snapshot)

            step = int(snapshot.split('_')[-1].split('.')[0]) / batch

            # some learning rate modify 

        else:

            init = tf.initialize_all_variables()

            sess.run(init)

            step = 0

        print '[*] start training ... '

        while step * batch < training_iters:

            #print '[*] step {} '.format(step)
            batch_x = []
            batch_y = []
            start = step * batch % len(images)
            end = (step + 1) * batch % len(images) # from zero

            if end < start:
                batch_x = np.vstack((images[start : ],images[:end]))
                batch_y = np.vstack((objects[start : ],objects[:end]))
            else:
                batch_x = images[start : end]
                batch_y = objects[start : end]

            
            assert batch_y.shape[-1] == int(y.get_shape()[-1])

            #print 'batch_y:',batch_y.shape
            #print 'batch_x:',batch_x.shape


            sess.run(optimizer, feed_dict = {
                                        x:batch_x,
                                        y:batch_y})

            

            if step % display_step == 0:

                cost = sess.run(loss,feed_dict = {
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
    n_width = 448
    n_height = 448
    n_input = n_width * n_height
    cls = ['plate']

    B = 2
    S = 7

    learning_rate = 1
    training_iters = 1500

    train(learning_rate, training_iters, batch, cls, display = 1, snapshot = args.snapshot)







