from utils import IoU
from skimage import io
import tensorflow as tf
from imdbs import load_imdb_from_raw
import numpy as np
from yolo_utils import cell_locate
#monkey.patch_all()

batch_size = 64
n_width = 448
n_height = 448
n_input = n_width * n_height
B = 2
S = 7
cls_name = ['plate','dog','mouse']
n_class = len(cls_name)

learning_rate = 0.1
training_iters = 1000

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

def conv_net(x, weights, biases, dropout):

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

weights = {

    'conv1': tf.Variable(tf.random_normal([7, 7, 3, 64])),

    'conv2': tf.Variable(tf.random_normal([3, 3, 64, 192])),

    'conv3': tf.Variable(tf.random_normal([1, 1, 192, 128])),
    'conv4': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    'conv5': tf.Variable(tf.random_normal([1, 1, 256, 256])),
    'conv6': tf.Variable(tf.random_normal([3, 3, 256, 512])),

    'conv7': tf.Variable(tf.random_normal([1, 1, 512, 256])),
    'conv8': tf.Variable(tf.random_normal([3, 3, 256, 512])),
    'conv9': tf.Variable(tf.random_normal([1, 1, 512, 256])),
    'conv10': tf.Variable(tf.random_normal([3, 3, 256, 512])),
    'conv11': tf.Variable(tf.random_normal([1, 1, 512, 256])),
    'conv12': tf.Variable(tf.random_normal([3, 3, 256, 512])),
    'conv13': tf.Variable(tf.random_normal([1, 1, 512, 256])),
    'conv14': tf.Variable(tf.random_normal([3, 3, 256, 512])),
    'conv15': tf.Variable(tf.random_normal([1, 1, 512, 512])),
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
    #'out':tf.Variable(tf.random_normal([]))
}

biases = {

    'conv1': tf.Variable(tf.random_normal([64])),

    'conv2': tf.Variable(tf.random_normal([192])),

    'conv3': tf.Variable(tf.random_normal([128])),
    'conv4': tf.Variable(tf.random_normal([256])),
    'conv5': tf.Variable(tf.random_normal([256])),
    'conv6': tf.Variable(tf.random_normal([512])),

    'conv7': tf.Variable(tf.random_normal([256])),
    'conv8': tf.Variable(tf.random_normal([512])),
    'conv9': tf.Variable(tf.random_normal([256])),
    'conv10': tf.Variable(tf.random_normal([512])),
    'conv11': tf.Variable(tf.random_normal([256])),
    'conv12': tf.Variable(tf.random_normal([512])),
    'conv13': tf.Variable(tf.random_normal([256])),
    'conv14': tf.Variable(tf.random_normal([512])),
    'conv15': tf.Variable(tf.random_normal([512])),
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


x = tf.placeholder(tf.float32, [None, n_input * 3]) # feed_dict (unknown batch , features)
y = tf.placeholder(tf.float32, [None, n_class + 4]) # feed_dict (unknown batch, prob for each classes)


is_obj = None
not_obj = None

response_threshold = 0.5

def get_confidence(pred, y, B):
    
    confidence = None

    print 'prediction : ', pred.get_shape()
    print 'ground truth : ', y.get_shape()
    
    for b in xrange(B):

        shape = (-1, int(pred.get_shape()[1]), b+1)    

        if type(confidence) is not tf.python.framework.ops.Tensor:

            confidence = IoU(tf.slice(pred,[0,0,b*5],[-1,-1,4]), y[:,:4])
        else:
            confidence = tf.concat(1,(confidence,IoU(tf.slice(pred,[0,0,b * 5],[-1,-1,4]), y[ : ,:4])))
            confidence = tf.reshape(confidence,shape)

    assert confidence.dtype == tf.float32


    return confidence



def convert_to_reality():

    pass

def responese(pred, gt):

    res = None

    for b in xrange(B):

        grid_cell_index = tf.cast(tf.reshape(cell_locate((n_width,n_height), gt, 7), [-1]), tf.int32)

        index = tf.range(0, batch_size)

        indices = tf.cast(tf.pack([index, grid_cell_index], axis = 1), tf.int64)

        res_sparse = tf.SparseTensor(indices = indices, values = tf.ones(batch_size), shape = [batch_size , S * S])


        if type(res) is not tf.python.framework.ops.Tensor:

            res = tf.sparse_tensor_to_dense(res_sparse)
        
        else:

            res = tf.concat(1,[res, tf.sparse_tensor_to_dense(res_sparse)])


    res = tf.reshape(res,[-1, S * S, B])
        

    print 'indices : ', indices.get_shape()


    print res.get_shape()


    """

    for b in xrange(B):
        
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


        gt_center_x = tf.mul(tf.add(pred_x, pred_w), 0.5)
        gt_center_y = tf.mul(tf.add(pred_y, pred_h), 0.5)




    """


def is_responsible(confidence):

    """
    threshold = max of confidence

    so is_res will be a boolean vector, present wheather the cell is responsible for the object

    """

    print 'confidence shape : ', confidence.get_shape()

    _, cells, B = list(confidence.get_shape())

    max_iou = tf.reduce_max(confidence, 2)
    
    for b in xrange(B-1):

        max_iou = tf.concat(1,[max_iou,max_iou])
    
    max_iou = tf.reshape(max_iou,[-1, int(cells), int(B)])
    is_res = tf.greater_equal(confidence, max_iou)

    assert is_res.dtype == bool and confidence.dtype == tf.float32
    return is_res

def is_appear_in_cell(confidence):

    return tf.greater(tf.reduce_sum(confidence,2),tf.zeros((batch_size, S * S)))

"""

training

"""


print 'load configurence '

lcoord = tf.constant(5, dtype = tf.float32)
lnoobj = tf.constant(0.5, dtype = tf.float32)

pred = conv_net(x, weights, biases, 1)

display_step = 1


confidence = get_confidence(pred, y, B)


test = responese(pred, y)

is_res = is_responsible(confidence)

is_appear = tf.cast(is_appear_in_cell(confidence), tf.float32)

not_res = tf.cast(tf.logical_not(is_res), tf.float32)

is_res = tf.cast(is_res, tf.float32)

images, objects = load_imdb_from_raw('5000_raw', cls_name)

images = np.array(images)

loss = None

for b in xrange(B):

    #print 'prediction : ', pred.get_shape()

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

    dx = tf.pow(tf.sub(pred_x, gt_x), 2)
    dy = tf.pow(tf.sub(pred_y, gt_y), 2)
    dw = tf.pow(tf.sub(tf.pow(pred_w,0.5), tf.pow(gt_w,0.5)), 2)
    dh = tf.pow(tf.sub(tf.pow(pred_h,0.5), tf.pow(gt_h,0.5)), 2)
    dc = tf.pow(tf.sub(pred_c, gt_c), 2)

    if loss == None:

        loss_coord_xy = tf.mul(tf.mul(lcoord, tf.slice(is_res,[0,0,b],[-1,-1,1])), tf.add(dx,dy))
        loss_coord_wh = tf.mul(tf.mul(lcoord, tf.slice(is_res,[0,0,b],[-1,-1,1])), tf.add(dw,dh))
        loss_is_obj = tf.mul(tf.slice(is_res,[0,0,b],[-1,-1,1]),dc)
        loss_no_obj = tf.mul(tf.slice(not_res,[0,0,b],[-1,-1,1]),dc)

  #      print 'is None loss : ', loss_coord_wh.get_shape()

        loss = tf.add(tf.add(loss_coord_xy,loss_coord_wh), tf.add(loss_is_obj,loss_no_obj))

    else:

        loss_coord_xy = tf.mul(tf.mul(lcoord, tf.slice(is_res,[0,0,b],[-1,-1,1])), tf.add(dx,dy))
        loss_coord_wh = tf.mul(tf.mul(lcoord, tf.slice(is_res,[0,0,b],[-1,-1,1])), tf.add(dw,dh))
        loss_is_obj = tf.mul(tf.slice(is_res,[0,0,b],[-1,-1,1]),dc)
        loss_no_obj = tf.mul(tf.slice(not_res,[0,0,b],[-1,-1,1]),dc)
 #       print 'is loss : ' ,loss_coord_wh.get_shape()
        loss = tf.add(loss, tf.add(tf.add(loss_coord_xy,loss_coord_wh), tf.add(loss_is_obj,loss_no_obj)))
#        print 'is loss is loss : ' ,loss.get_shape()

    index = b + 1


"""
reshape loss [batch, cell, bbox] to [batch, bbox], so we can sum over all bbox

"""

loss = tf.reshape(tf.reduce_sum(loss,1), [-1])

y_cls = tf.reshape(tf.slice(pred, [0,0,5 * index], [-1,-1,-1]),[-1, S * S, n_class])

pred_cls = tf.slice(pred, [0,0,5 * index], [-1,-1,-1])

gt_cls = tf.pow(tf.sub(y_cls, pred_cls), 2)

#print 'gt_cls : ', gt_cls.get_shape()

is_appear = tf.reshape(is_appear, [-1, S * S, 1])
gt_cls = tf.mul(is_appear, gt_cls)
gt_cls = tf.reduce_sum(gt_cls,1)
gt_cls = tf.reduce_sum(gt_cls,1)
loss = tf.add(loss, gt_cls)

assert int(tf.slice(y,[0,4],[-1,-1]).get_shape()[1]) == n_class
loss = tf.reduce_mean(loss)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

init = tf.initialize_all_variables()


with tf.Session() as sess:
    sess.run(init)
    step = 0

    print 'start training ... '

    while step * batch_size < training_iters:

        batch_x = []
        batch_y = []

        start = step * batch_size % len(images)
        end = (step + 1) * batch_size % len(images) # from zero

        if end < start:
            batch_x = np.vstack((images[start : ],images[:end]))
            batch_y = np.vstack((objects[start : ],objects[:end]))
        else:
            batch_x = images[start : end]
            batch_y = objects[start : end]

        
        assert batch_y.shape[-1] == int(y.get_shape()[-1])

        print 'batch_y:',batch_y.shape
        print 'batch_x:',batch_x.shape


        sess.run(optimizer, feed_dict = {
                                    x:batch_x,
                                    y:batch_y})

        print 'step {} '.format(step)
        if step % display_step == 0:

            cost = sess.run([loss],
                            feed_dict = {
                                x:batch_x,
                                y:batch_y})

            print "Iter " , str(step*batch_size) + ", Minibatch Loss = " , cost
            #, Training Accuracy= " ,"{:.5f}".format(acc)
        step += 1

    print "Optimization Finished!"
