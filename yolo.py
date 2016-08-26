from yolo_utils import cell_locate, convert_to_one, convert_to_reality
from imdbs import load_imdb_from_raw
from utils import IoU
from skimage import io
import tensorflow as tf
import numpy as np
import argparse
import skimage
import sys
import cv2
def draw(bbox, image, label, iou_threshold):

    for i, b in enumerate(bbox):

        #print b[4]
        if b[4] > iou_threshold :

            x1 = int(b[0])
            y1 = int(b[1])

            x2 = 448 if int(b[0] + b[2]) > 448 else int(b[0] + b[2])
            y2 = 448 if int(b[1] + b[3]) > 448 else int(b[1] + b[3])

            assert len(b[5:]) == 35
            #print b[5:]
            #print np.argmax(b[5:])

            cls = label[np.argmax(b[5:])]

            cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 2)



            print '[ {} {} {} {} ] [ {} ] [ {} ]'.format(x1, y1, x2, y2, b[4], cls)

    cv2.imshow('image', image)
    cv2.waitKey(0)


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




def conv2d(x, W, b, strides = 1, norm = False):

    # input [ batch , height , width , channels ]
    # filters [ width , height , channels , output channels (number of filters)]
    x = tf.nn.conv2d(x, W, strides = [1, strides, strides, 1], padding = 'SAME')
    x = tf.nn.bias_add(x, b)

    #if norm: 
        #x = tf.nn.l2_normalize(x, 1, epsilon=1e-12, name=None)

    x = leakey(x)
    x = tf.nn.local_response_normalization(x)

    return x

def one(x):

    maximum = tf.reshape(tf.reduce_max(x, 1), [-1, 1])
    minimum = tf.reshape(tf.reduce_min(x, 1), [-1, 1])
    return (x - minimum) / (maximum - minimum)

def maxpool2d(x, k = 2):
    return tf.nn.max_pool(x, ksize = [1, k, k, 1], strides = [1, k, k, 1], padding = 'SAME')
def avgpool2d(x, k = 2):
    return tf.nn.avg_pool(x, ksize = [1, k, k, 1], strides = [1, k, k, 1], padding = 'SAME')

def leakey(conv_node):

    return tf.maximum(tf.mul(0.1, conv_node), conv_node)
    #return tf.nn.relu(conv_node)

def log(tensor, string):

    return tf.Print(tensor, [tensor], string)

def conv_net(x, weights, biases, n_class, dropout):

    x = tf.reshape(x, shape = [-1, 448, 448, 3])

    # Convolution Layer
 
    conv1 = conv2d(x, weights['conv1'], biases['conv1'], strides = 1, norm = False)
    #conv1 = tf.nn.l2_normalize(conv1, 1, epsilon=1e-12, name=None)
    conv1 = maxpool2d(conv1, k = 2)
    #conv1 = log(conv1, 'conv1 : ')

    conv2 = conv2d(conv1, weights['conv2'], biases['conv2'], strides = 1, norm = False)
    #conv2 = tf.nn.l2_normalize(conv2, 1, epsilon=1e-12, name=None)
    conv2 = maxpool2d(conv2, k = 2)

    conv3 = conv2d(conv2, weights['conv3'], biases['conv3'], strides = 1, norm = False)
    #conv3 = tf.nn.l2_normalize(conv3, 1, epsilon=1e-12, name=None)
    conv3 = maxpool2d(conv3, k = 2)

    conv4 = conv2d(conv3, weights['conv4'], biases['conv4'], strides = 1, norm = False)
    #conv4 = tf.nn.l2_normalize(conv4, 1, epsilon=1e-12, name=None)
    conv4 = maxpool2d(conv4, k = 2)

    conv5 = conv2d(conv4, weights['conv5'], biases['conv5'], strides = 1, norm = False)
    #conv5 = tf.nn.l2_normalize(conv5, 1, epsilon=1e-12, name=None)
    conv5 = maxpool2d(conv5, k = 2)

    conv6 = conv2d(conv5, weights['conv6'], biases['conv6'], strides = 1, norm = False)
    #conv6 = tf.nn.l2_normalize(conv6, 1, epsilon=1e-12, name=None)
    conv6 = maxpool2d(conv6, k = 2)

    conv7 = conv2d(conv6, weights['conv7'], biases['conv7'], strides = 1, norm = False)
    #conv7 = tf.nn.l2_normalize(conv7, 1, epsilon=1e-12, name=None)
    conv8 = conv2d(conv7, weights['conv8'], biases['conv8'], strides = 1, norm = False)
    #conv8 = tf.nn.l2_normalize(conv8, 1, epsilon=1e-12, name=None)
    conv9 = conv2d(conv8, weights['conv9'], biases['conv9'], strides = 1, norm = False)
    #conv9 = tf.nn.l2_normalize(conv9, 1, epsilon=1e-12, name=None)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    #fc1 = tf.reshape(conv9, [-1, 256])

    #fc1 = tf.transpose(conv9,(0,3,1,2))
   
    fc1 = tf.reshape(conv9, [-1, weights['fc1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['fc1']), biases['fc1'])
    fc1 = leakey(fc1)
    fc1 = tf.nn.l2_normalize(fc1, 1, epsilon=1e-12, name=None)
    
    #fc1 = tf.nn.local_response_normalization(fc1)

    #fc1 = tf.nn.relu(fc1)
   
    # fc1 = tf.nn.dropout(fc1, dropout)

    fc2 = tf.add(tf.matmul(fc1, weights['fc2']), biases['fc2'])
    fc2 = leakey(fc2)
    fc2 = tf.nn.l2_normalize(fc2, 1, epsilon=1e-12, name=None)
    
    #fc2 = tf.nn.local_response_normalization(fc2)

    #fc2 = log(fc2, 'fc2 : ')
    #fc2 = tf.reshape(fc2, [-1, 4096])
    #fc2 = tf.nn.relu(fc2)
    
    fc3 = tf.add(tf.matmul(fc2, weights['fc3']), biases['fc3'])
    fc3 = one(fc3)
    fc3 = tf.reshape(fc3, [-1, S * S, n_class + 5 * B])
    #fc3 = tf.nn.l2_normalize(fc3, 1, epsilon=1e-12, name=None)
    
    #fc3 = tf.abs(fc3)
    #fc3 = log(fc3, 'fc3 : ')

    return fc3

def init(session, saver, pretrained):

    pass

def save(session, saver, prefix, step):

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

        pred_x = log(pred_x, 'pred one bbox x : ')
        pred_y = log(pred_y, 'pred one bbox y : ')
        pred_w = log(pred_w, 'pred one bbox w : ')
        pred_h = log(pred_h, 'pred one bbox h : ')
   
        pred_bbox = [pred_x, pred_y, pred_w, pred_h]

        pred_reality_bbox = convert_to_reality(pred_bbox, n_width, n_height, S)
        pred_reality_bbox[0] = log(pred_reality_bbox[0], 'pred bbox x : ')
        pred_reality_bbox[1] = log(pred_reality_bbox[1], 'pred bbox y : ')
        pred_reality_bbox[2] = log(pred_reality_bbox[2], 'pred bbox w : ')
        pred_reality_bbox[3] = log(pred_reality_bbox[3], 'pred bbox h : ')

        temp = tf.reshape(IoU(pred_reality_bbox, y[:,:4]), [-1, S * S, 1])

        if type(confidence) is not tf.python.framework.ops.Tensor:

            confidence = temp

        else:
     
            confidence = tf.concat(2, [confidence, temp])

    assert confidence.dtype == tf.float32

    return confidence

def Center(groundtruth, batch):

    """

    check which cell is the center of the object located


    """

    c = None

    #for b in xrange(B):

        # grid cell index tensor shape : [ batch , cells , one of bboxes ]

        # find out which grid cell is the bbox center located in

    grid_cell_index = tf.cast(tf.reshape(cell_locate((n_width, n_height), groundtruth, S), [-1]), tf.int32)

        # generate index for terrible tensorflow slicing tensor

    index = tf.range(0, batch)

        # pack index with grid cell index 

    indices = tf.cast(tf.pack([index, grid_cell_index], axis = 1), tf.int64)

        # set the " center variable " to one ( which is boolean type ), in terms of grid cell index  

    temp = tf.SparseTensor(indices = indices, values = tf.ones(batch), shape = [batch , S * S])
    center_sparse = tf.sparse_tensor_to_dense(temp)
    c = tf.reshape(center_sparse, [-1, S * S, 1])
        
        # center_sparse = tf.reshape(center_sparse, [-1, S * S, 1])

        # convert sparse tensor to dense tensor, and set others to 0 , represent that they are no responsible to the object

    """
        if type(c) is not tf.python.framework.ops.Tensor:

            c = tf.reshape(center_sparse, [-1, S * S, 1])
          
        else:
           
            center_sparse = tf.reshape(center_sparse, [-1, S * S, 1])

            c = tf.concat(2, [c, center_sparse])
    """

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

    #iou = tf.mul(center, confidence)

    iou = tf.reshape(confidence, [-1, S * S * B])
    #iou = tf.reshape(tf.mul(center, confidence), [-1, S * S * B])
    # find out maximum IoU

    #maximum_IoU = tf.reshape(tf.reduce_max(iou, 2), [-1, S * S, 1])


    # reshape 49 x 2 to 98 , reduce max find the maximum value

    maximum_IoU = tf.reshape(tf.reduce_max(iou, 1), [-1, 1])

    # Create a same shape of tensor as iou 

    #_iou = tf.concat(2, [maximum_IoU, maximum_IoU])

    #for b in xrange(B - 2):

        #_iou = tf.concat(2, [_iou, maximum_IoU])

    #maximum_IoU = _iou



    # return the bool type tensor 

    res = tf.logical_and(tf.greater(iou, 0),tf.greater_equal(iou, maximum_IoU))

    res = tf.reshape(res, [-1, S * S, B])

    return res

def Appear(confidence, batch):

    return tf.greater(tf.reduce_sum(confidence, 2),tf.zeros((batch, S * S)))

def softmax(logits):

    e = tf.exp(logits)

    return e / tf.reshape(tf.reduce_sum(e, 2), [-1, 49, 1])




"""

training

"""


def train(learning_rate, iters, batch, label, dataset, n_bbox = 2, n_cell = 7, n_width = 448, n_height = 448, display = 20, threshold = 0.5, snapshot = None, pretrained_weights = None):
    
    print '[*] loading configurence '

    n_class = len(label)

    # load dataset

    images, objects, filename = load_imdb_from_raw(dataset, label, batch)

    #images = np.array(images)

    x = tf.placeholder(tf.float32, [None, n_input * 3]) # feed_dict (unknown batch, features)
    y = tf.placeholder(tf.float32, [None, n_class + 4]) # feed_dict (unknown batch, prob for each classes)
  
    is_obj = None
    not_obj = None
    loss = None

    response_threshold = threshold
    display_step = display

    weights = {

        
        'conv1': tf.Variable(tf.random_normal([3, 3, 3, 16], mean = 0, stddev = 0.5), name = 'w_conv1'),
        'conv2': tf.Variable(tf.random_normal([3, 3, 16, 32], mean = 0, stddev = 0.5), name = 'w_conv2'),
        'conv3': tf.Variable(tf.random_normal([3, 3, 32, 64], mean = 0, stddev = 0.5), name = 'w_conv3'),
        'conv4': tf.Variable(tf.random_normal([3, 3, 64, 128], mean = 0, stddev = 0.5), name = 'w_conv4'),
        'conv5': tf.Variable(tf.random_normal([3, 3, 128, 256], mean = 0, stddev = 0.5), name = 'w_conv5'),
        'conv6': tf.Variable(tf.random_normal([3, 3, 256, 512], mean = 0, stddev = 0.5), name = 'w_conv6'),
        'conv7': tf.Variable(tf.random_normal([3, 3, 512, 1024], mean = 0, stddev = 0.5), name = 'w_conv7'),
        'conv8': tf.Variable(tf.random_normal([3, 3, 1024, 1024], mean = 0, stddev = 0.5), name = 'w_conv8'),
        'conv9': tf.Variable(tf.random_normal([3, 3, 1024, 1024], mean = 0, stddev = 0.5), name = 'w_conv9'),

        'fc1':tf.Variable(tf.random_normal([7 * 7 * 1024, 256], mean = 0, stddev = 0.5), name = 'w_fc1'),
        'fc2':tf.Variable(tf.random_normal([256, 4096], mean = 0, stddev = 0.5), name = 'w_fc2'),
        'fc3':tf.Variable(tf.random_normal([4096, 7 * 7 * (n_class + 5 * B)], mean = 0, stddev = 0.5), name = 'w_fc3'),
     
    }

    biases = {

        'conv1': tf.Variable(tf.random_normal([16], mean = 0, stddev = 0.5), name = 'b_conv1'),
        'conv2': tf.Variable(tf.random_normal([32], mean = 0, stddev = 0.5), name = 'b_conv2'),
        'conv3': tf.Variable(tf.random_normal([64], mean = 0, stddev = 0.5), name = 'b_conv3'),
        'conv4': tf.Variable(tf.random_normal([128], mean = 0, stddev = 0.5), name = 'b_conv4'),
        'conv5': tf.Variable(tf.random_normal([256], mean = 0, stddev = 0.5), name = 'b_conv5'),
        'conv6': tf.Variable(tf.random_normal([512], mean = 0, stddev = 0.5), name = 'b_conv6'),
        'conv7': tf.Variable(tf.random_normal([1024], mean = 0, stddev = 0.5), name = 'b_conv7'),
        'conv8': tf.Variable(tf.random_normal([1024], mean = 0, stddev = 0.5), name = 'b_conv8'),
        'conv9': tf.Variable(tf.random_normal([1024], mean = 0, stddev = 0.5), name = 'b_conv9'),
        
        'fc1':tf.Variable(tf.random_normal([256], mean = 0, stddev = 0.5), 'b_fc1'),
        'fc2':tf.Variable(tf.random_normal([4096], mean = 0, stddev = 0.5), 'b_fc2'),
        'fc3':tf.Variable(tf.random_normal([S * S * (n_class + 5 * B)], mean = 0, stddev = 0.5), 'b_fc3'),

    }
    
    lcoord = tf.constant(5, dtype = tf.float32)
    lnoobj = tf.constant(0.5, dtype = tf.float32)
    #lcoord2 = tf.constant(1, dtype = tf.float32)
    #lnoobj2 = tf.constant(1, dtype = tf.float32)
    # forward propagate

    pred = conv_net(x, weights, biases, n_class, 1)
    
    confidence = Confidence(pred, y)
    confidence = log(confidence, 'confidence : ')

    center = Center(y, batch) # training only wants one bbox , but in test time , use nms to iliminate other bbox
    #center = log(center, 'center : ')
    
   
    responsible = Responsible(center, confidence)
    #responsible = log(responsible, 'responsible : ')

    #appear = tf.cast(Appear(confidence, batch), tf.float32)


    not_responsible = tf.cast(tf.logical_not(responsible), tf.float32)
    responsible = tf.cast(responsible, tf.float32) 

    #responsible = log(responsible, 'responsible 2 : ') 
    #not_responsible = log(not_responsible, 'not responsible 2 : ') 

    # create loss function


    gt_x = tf.reshape(tf.slice(y, [0, 0], [-1, 1]), [-1, 1, 1])
    gt_y = tf.reshape(tf.slice(y, [0, 1], [-1, 1]), [-1, 1, 1]) 
    gt_w = tf.reshape(tf.slice(y, [0, 2], [-1, 1]), [-1, 1, 1])
    gt_h = tf.reshape(tf.slice(y, [0, 3], [-1, 1]), [-1, 1, 1])
    bbox = [gt_x, gt_y, gt_w, gt_h]

    # convert the x , y to the offset of grid cells and w , h to the range of 0 to 1 by divided by image size

    # offset x , offset y will only located in one grid cells , others will be set to 0 by the variable " Responsible "
 
    gt_offset_x, gt_offset_y, gt_offset_w, gt_offset_h = convert_to_one(bbox, n_width, n_height, S)
    r_loss_coord_xy = None
    for b in xrange(B):

        pred_offset_x = tf.slice(pred, [0, 0, b * 5 + 0], [-1, -1, 1])
        pred_offset_y = tf.slice(pred, [0, 0, b * 5 + 1], [-1, -1, 1])
        pred_offset_w = tf.slice(pred, [0, 0, b * 5 + 2], [-1, -1, 1])
        pred_offset_h = tf.slice(pred, [0, 0, b * 5 + 3], [-1, -1, 1])
        pred_confidence = tf.slice(pred, [0, 0, b * 5 + 4], [-1, -1, 1])


        """
            pred_x = log(pred_x, 'pred  X  : ')
            pred_y = log(pred_y, 'pred  Y  : ')
            pred_w = log(pred_w, 'pred  W  : ')
            pred_h = log(pred_h, 'pred  H  : ')
            pred_c = log(pred_c, 'pred  C  : ')

        """

        
        gt_confidence = tf.slice(confidence, [0, 0, b], [-1, -1, 1])


            # predict confidence and groundtruth confidence, so the predictor can learn if it contains a objects


        d_offset_x = tf.pow(tf.sub(tf.cast(pred_offset_x, tf.float32), tf.cast(gt_offset_x, tf.float32)), 2)
        d_offset_y = tf.pow(tf.sub(tf.cast(pred_offset_y, tf.float32), tf.cast(gt_offset_y, tf.float32)), 2)
        d_offset_w = tf.pow(tf.sub(tf.cast(tf.pow(tf.abs(pred_offset_w), 0.5), tf.float32), tf.cast(tf.pow(tf.abs(gt_offset_w), 0.5), tf.float32)), 2)
        d_offset_h = tf.pow(tf.sub(tf.cast(tf.pow(tf.abs(pred_offset_h), 0.5), tf.float32), tf.cast(tf.pow(tf.abs(gt_offset_h), 0.5), tf.float32)), 2)
        d_confidence = tf.pow(tf.sub(pred_confidence, gt_confidence), 2) 
        
        """
        dx = log(dx, 'dx : ')
        dy = log(dy, 'dy : ')
        dc = log(dc, 'dc : ')
        dw = log(dw, 'dw : ')
        """

        loss_coord_xy = tf.mul(tf.mul(lcoord, tf.slice(responsible,[0, 0, b],[-1,-1, 1])), tf.concat(2, [d_offset_x, d_offset_y]))     
        loss_coord_wh = tf.mul(tf.mul(lcoord, tf.slice(responsible,[0, 0, b],[-1,-1, 1])), tf.concat(2, [d_offset_w, d_offset_h]))

        """
        " reality box loss "

        pred_offset_bbox = [pred_offset_x, pred_offset_y, pred_offset_w, pred_offset_h]
        pred_offset_bbox = convert_to_reality(pred_offset_bbox, n_width, n_height, S)

        pred_real_x = tf.cast(pred_offset_bbox[0], tf.float32)
        pred_real_y = tf.cast(pred_offset_bbox[1], tf.float32)

        pred_real_w = tf.cast(pred_offset_bbox[2], tf.float32)

        pred_real_h = tf.cast(pred_offset_bbox[3], tf.float32)


        d_real_x = tf.pow(tf.sub(pred_real_x, tf.cast(gt_x, tf.float32)), 2)
        d_real_y = tf.pow(tf.sub(pred_real_y, tf.cast(gt_y, tf.float32)), 2)
        d_real_w = tf.pow(tf.sub(tf.cast(tf.pow(tf.abs(pred_real_w), 0.5), tf.float32), tf.cast(tf.pow(tf.abs(gt_w), 0.5), tf.float32)), 2)
        d_real_h = tf.pow(tf.sub(tf.cast(tf.pow(tf.abs(pred_real_h), 0.5), tf.float32), tf.cast(tf.pow(tf.abs(gt_h), 0.5), tf.float32)), 2)

        loss_real_xy = tf.mul(tf.mul(lcoord2 , tf.slice(responsible,[0, 0, b],[-1,-1, 1])), tf.concat(2, [d_real_x, d_real_y]))     
        loss_real_wh = tf.mul(tf.mul(lcoord2 , tf.slice(responsible,[0, 0, b],[-1,-1, 1])), tf.concat(2, [d_real_w, d_real_h]))
        """
        
        loss_is_obj = tf.mul(tf.slice(responsible, [0, 0, b], [-1, -1, 1]), d_confidence)
        loss_no_obj = tf.mul(tf.mul(tf.slice(not_responsible, [0, 0, b], [-1,-1, 1]), d_confidence), lnoobj)

        #loss_coord_xy = log(tf.reduce_mean(loss_coord_xy), "loss xy : ")
        #loss_coord_wh = log(tf.reduce_mean(loss_coord_wh), "loss wh : ")
        #loss_is_obj = log(loss_is_obj, "loss is obj : ")
        #loss_no_obj = log(loss_no_obj, "loss no obj : ")



        if loss == None:

            loss = tf.concat(2, [tf.concat(2, [loss_coord_xy, loss_coord_wh]), tf.add(loss_is_obj, loss_no_obj)])

        else:

            loss = tf.concat(2, [loss, tf.concat(2, [tf.concat(2, [loss_coord_xy, loss_coord_wh]), tf.add(loss_is_obj, loss_no_obj)])])

        index = b + 1

    
    # reshape loss [batch, cell, bbox] to [batch, bbox], so we can sum over all bbox

    #gt_cls = tf.reshape(tf.slice(y, [0, 0, 5 * index], [-1,-1,-1]), [-1, S * S, n_class])

    gt_cls = tf.reshape(y[ : , 4: ], [-1, 1, n_class])

    gt_cls = log(gt_cls, 'gt cls : ')


    pred_cls = tf.slice(pred, [0, 0, 5 * index], [-1,-1,-1])

    pred_cls = log(pred_cls, 'pred cls : ')

    dcls = tf.pow(tf.sub(pred_cls, gt_cls), 2)

    #dcls = tf.sub(softmax(pred_cls), gt_cls)

    dcls = tf.reshape(tf.reduce_sum(dcls, 2), [-1, S * S, 1])

    dcls = log(dcls, "d cls 1 : ")

    #dcls = tf.mul(tf.slice(center, [0, 0, 0], [-1,-1, 1]), dcls)
    dcls = tf.mul(center, dcls)

    #dcls = log(dcls, "d cls 2 : ")

    loss = tf.concat(2, [loss, dcls])

    assert int(tf.slice(y,[0, 4],[-1,-1]).get_shape()[1]) == n_class

    loss = tf.reduce_mean(loss)

    #loss = log(loss, " loss : ")

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, aggregation_method = tf.AggregationMethod.EXPERIMENTAL_TREE)

    #correct_pred = tf.equal(tf.argmax(pred_cls, 2), tf.argmax(gt_cls, 2))
    #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.Session() as sess:

        convolution_weights = [
                                     weights['conv1'],
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
                                     biases['conv9']
                                ]
      
        fully_connect_weights = [   
                                    weights['fc1'],
                                    weights['fc2'],
                                    weights['fc3'],
                                    biases['fc1'],
                                    biases['fc2'],
                                    biases['fc3']
                                ]

        if pretrained_weights is not None:

            #print 'pretrained weight : ', pretrained_weights
            print '[*] Loading pretrained weights : {}'.format(pretrained_weights) 
            load_convolution_weights = tf.train.Saver(convolution_weights)
            tf.initialize_all_variables().run()
            load_convolution_weights.restore(sess, pretrained_weights)
            step = 0
            # some learning rate modify 

        elif snapshot is not None:

            print '[*] Loading snapshot : {}'.format(snapshot)

            tf.initialize_all_variables().run()    
            load_total_weights = tf.train.Saver()        
            load_total_weights.restore(sess, snapshot)
            step = int(snapshot.split('_')[-1].split('.')[0]) / batch          

        else:

            # train from begin
            saver = tf.train.Saver()
            init = tf.initialize_all_variables()
            sess.run(init)
            step = 0

        print '[*] Start Training ... '

        while step * batch < training_iters:

            batch_x = images.next_batch()
            batch_y = objects.next_batch()

            learning_rate = tf.train.exponential_decay(
                                0.00001,                # Base learning rate.
                                step * batch,  # Current index into the dataset.
                                10000,          # Decay step.
                                0.9,                # Decay rate.
                                staircase=True)

            

            if step % display_step == 0:

                print '-----------------------------------------------------------------'

                cost, l_rate, res, conf, _dcls = sess.run([tf.reduce_mean(loss), learning_rate, responsible, confidence, dcls], { x : batch_x , y : batch_y })

                print "Iter " , str(step * batch)  + " , Minibatch Loss = " , cost ," , Learning rate : ", l_rate

                #print 'resonsible : ', res[0]
                #print 'confidence : ', conf[0]
                #print 'conf : ', res[0] * conf[0]
            
                #print 'conf : ', conf.shape
                #print 'dcls : ', _dcls[0]
                #print 'dcls shape : ', _dcls.shape

            
            sess.run(optimizer, { x : batch_x, y : batch_y })

            step += 1

            

        yolo_tiny = convolution_weights + fully_connect_weights

        saver = tf.train.Saver(yolo_tiny)

        snapshot_saver = tf.train.Saver()

        p = save(sess, saver,'./char_grad', str(step * batch))
        p = save(sess, snapshot_saver,'./char_grad_snapshot', str(step * batch))

        print '[*] Model saved in file : {}'.format(p)

        print "[*] Optimization Finished!"




if __name__ == '__main__':

    args = parse_args()

    batch = 50
    display = 1
    #dataset = 'plate_300'
    dataset = 'char'
    n_width = 448
    n_height = 448
    n_input = n_width * n_height

    #label = ['plate']
    label = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z', '0','1','2','3','4','5','6','7','8','9']

    n_class = len(label)

    B = 2
    S = 7

    
    

    if args.mode == 'train':

        
        learning_rate = 0.00001
        training_iters = 20000
        train(learning_rate, training_iters, batch, label, dataset, display = 1, snapshot = args.snapshot, pretrained_weights = args.weights)
    
    elif args.mode == 'test':

        x = tf.placeholder(tf.float32, [n_input * 3])
        image = io.imread(args.image, False)
        image = skimage.img_as_float(image)
        image = cv2.resize(image, (448, 448))

        data = image.flatten()
        weights = {

        
        'conv1': tf.Variable(tf.random_normal([3, 3, 3, 16], mean = 0, stddev = 0.5), name = 'w_conv1'),
        'conv2': tf.Variable(tf.random_normal([3, 3, 16, 32], mean = 0, stddev = 0.5), name = 'w_conv2'),
        'conv3': tf.Variable(tf.random_normal([3, 3, 32, 64], mean = 0, stddev = 0.5), name = 'w_conv3'),
        'conv4': tf.Variable(tf.random_normal([3, 3, 64, 128], mean = 0, stddev = 0.5), name = 'w_conv4'),
        'conv5': tf.Variable(tf.random_normal([3, 3, 128, 256], mean = 0, stddev = 0.5), name = 'w_conv5'),
        'conv6': tf.Variable(tf.random_normal([3, 3, 256, 512], mean = 0, stddev = 0.5), name = 'w_conv6'),
        'conv7': tf.Variable(tf.random_normal([3, 3, 512, 1024], mean = 0, stddev = 0.5), name = 'w_conv7'),
        'conv8': tf.Variable(tf.random_normal([3, 3, 1024, 1024], mean = 0, stddev = 0.5), name = 'w_conv8'),
        'conv9': tf.Variable(tf.random_normal([3, 3, 1024, 1024], mean = 0, stddev = 0.5), name = 'w_conv9'),

        'fc1':tf.Variable(tf.random_normal([7 * 7 * 1024, 256], mean = 0, stddev = 0.5), name = 'w_fc1'),
        'fc2':tf.Variable(tf.random_normal([256, 4096], mean = 0, stddev = 0.5), name = 'w_fc2'),
        'fc3':tf.Variable(tf.random_normal([4096, 7 * 7 * (n_class + 5 * B)], mean = 0, stddev = 0.5), name = 'w_fc3'),
     
        }

        biases = {

        'conv1': tf.Variable(tf.random_normal([16], mean = 0, stddev = 0.5), name = 'b_conv1'),
        'conv2': tf.Variable(tf.random_normal([32], mean = 0, stddev = 0.5), name = 'b_conv2'),
        'conv3': tf.Variable(tf.random_normal([64], mean = 0, stddev = 0.5), name = 'b_conv3'),
        'conv4': tf.Variable(tf.random_normal([128], mean = 0, stddev = 0.5), name = 'b_conv4'),
        'conv5': tf.Variable(tf.random_normal([256], mean = 0, stddev = 0.5), name = 'b_conv5'),
        'conv6': tf.Variable(tf.random_normal([512], mean = 0, stddev = 0.5), name = 'b_conv6'),
        'conv7': tf.Variable(tf.random_normal([1024], mean = 0, stddev = 0.5), name = 'b_conv7'),
        'conv8': tf.Variable(tf.random_normal([1024], mean = 0, stddev = 0.5), name = 'b_conv8'),
        'conv9': tf.Variable(tf.random_normal([1024], mean = 0, stddev = 0.5), name = 'b_conv9'),
        
        'fc1':tf.Variable(tf.random_normal([256], mean = 0, stddev = 0.5), 'b_fc1'),
        'fc2':tf.Variable(tf.random_normal([4096], mean = 0, stddev = 0.5), 'b_fc2'),
        'fc3':tf.Variable(tf.random_normal([S * S * (n_class + 5 * B)], mean = 0, stddev = 0.5), 'b_fc3'),

        }
    
        with tf.device('/cpu:0'):

            pred = conv_net(x, weights, biases, n_class, 1)

        print n_class

        with tf.Session() as sess:

            if args.weights is not None:

                convolution_weights = [
                         weights['conv1'],
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
                         biases['conv9']]
      
                fully_connect_weights = [ 
                            weights['fc1'],
                            weights['fc2'],
                            weights['fc3'],
                            biases['fc1'],
                            biases['fc2'],
                            biases['fc3']]

                tf.initialize_all_variables().run()

                saver = tf.train.Saver(convolution_weights + fully_connect_weights)

                saver.restore(sess, args.weights)

                print '[*] load weights from : {}'.format(args.weights)

            else:

                print 'required weights for testing ... '

                raise

            p = sess.run(pred, { x : data })

            for b in xrange(B):

                pred_x = p[:,:,b*5+0]
                pred_y = p[:,:,b*5+1]
                pred_w = p[:,:,b*5+2]
                pred_h = p[:,:,b*5+3]
                pred_c = p[:,:,b*5+4]
                pred_cls = p[:,:,B*5:]
               
                pred_bbox = [pred_x, pred_y, pred_w, pred_h]

                print ' predict : {} {} {} {}'.format(pred_x[0,0], pred_y[0,0], pred_w[0,0], pred_h[0,0])

                x, y, w, h = sess.run(convert_to_reality(pred_bbox, n_width, n_height, S))

                x = np.reshape(x, (49, 1))
                y = np.reshape(y, (49, 1))
                w = np.reshape(w, (49, 1))
                h = np.reshape(h, (49, 1))
                c = np.reshape(pred_c, (49, 1))

                #print 'class : ', n_class

               # print 'pred : ', pred_cls.shape

                pred_cls = np.reshape(pred_cls, (49, n_class))
                #print 'pred cls ; ', pred_cls.shape
                bbox = np.hstack((x, y, w, h, c, pred_cls))
                draw(bbox, image, label, 0.7)