from utils import IoU
from skimage import io
import tensorflow as tf
from imdb import load_imdb
import numpy as np


class cfg(object):

    def __init__(self, cls_name):
        self.cls_num = len(cls_name)
        self.cls_name = cls_name


#plate_cfg = cfg(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','plate'])

batch_size = 5
n_input = 448 * 448
B = 2
S = 7
cls_name = ['plate','dog']
n_class = len(cls_name)

learning_rate = 0.01
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

    #print weights['conv21'].get_shape()
    #print x.get_shape()
    #print biases['conv21'].get_shape()
    conv1 = conv2d(x, weights['conv1'], biases['conv1'], strides = 2)
    conv1 = maxpool2d(conv1, k = 2)

    conv2 = conv2d(conv1, weights['conv2'], biases['conv2'])
    conv2 = maxpool2d(conv2, k = 2)

    conv3 = conv2d(conv2, weights['conv3'], biases['conv3'])
    conv4 = conv2d(conv3, weights['conv4'], biases['conv4'])
    conv5 = conv2d(conv4, weights['conv5'], biases['conv5'])
    conv6 = conv2d(conv5, weights['conv6'], biases['conv6'])
    conv6 = maxpool2d(conv6, k = 2)

    conv7 = conv2d(conv6, weights['conv7'], biases['conv7'])
    conv8 = conv2d(conv7, weights['conv8'], biases['conv8'])
    conv9 = conv2d(conv8, weights['conv9'], biases['conv9'])
    conv10 = conv2d(conv9, weights['conv10'], biases['conv10'])
    conv11 = conv2d(conv10, weights['conv11'], biases['conv11'])
    conv12 = conv2d(conv11, weights['conv12'], biases['conv12'])
    conv13 = conv2d(conv12, weights['conv13'], biases['conv13'])
    conv14 = conv2d(conv13, weights['conv14'], biases['conv14'])
    conv15 = conv2d(conv14, weights['conv15'], biases['conv15'])
    conv16 = conv2d(conv15, weights['conv16'], biases['conv16'])
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
    print 'conv7  : ', conv7.get_shape()
    print 'conv8  : ', conv8.get_shape()
    print 'conv9  : ', conv9.get_shape()
    print 'conv10  : ', conv10.get_shape()
    print 'conv11  : ', conv11.get_shape()
    print 'conv12  : ', conv12.get_shape()
    print 'conv13  : ', conv13.get_shape()
    print 'conv14  : ', conv14.get_shape()
    print 'conv15  : ', conv15.get_shape()
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

#def confidence_equal_zero():
#    print 'zeros'
#    return tf.constant(11)
    #return IoU(pred[ : , : , b * 5 : b * 5 + 4], y[ : , b * 5 : b * 5 + 4])

#def confidence_not_equal_zero():
#    print 'not zeros ...'
#    return tf.constant(2)
    #return tf.concat(1,(confidence,IoU(pred[ : , : , b * 5 : b * 5 + 4], y[ : , b * 5 : b * 5 + 4])))
def get_confidence(pred, y, B):

    #confidence = tf.Variable(np.array([]), tf.float32)

    #zero = tf.constant(np.array([]))

    #print 'pred_shape : ', pred.get_shape()[1]
    shape = (5,int(pred.get_shape()[1]),B)

    """
    #print shape
    #   for b in xrange(B):
        print 'B : ', b
        #print 'pred : ', pred[ : , : , b * 5 : b * 5 + 4].get_shape()
        #xx = tf.constant(1)
        #yy = tf.constant(2)
        #print 'true or not ' , xx < yy
        print 'tf.test 0 : ', IoU(pred[ : , : , b * 5 : b * 5 + 4], y[ : , b * 5 : b * 5 + 4]).get_shape()
        print 'con ' , confidence.dtype
        print 'confid : ', confidence.get_shape()
        #print 'tf.test 1 : ', tf.concat(1,(confidence,IoU(pred[ : , : , b * 5 : b * 5 + 4], y[ : , b * 5 : b * 5 + 4]))).get_shape()
        #confidence = tf.cond(
                #tf.equal(confidence, zero),
                #confidence_equal_zero,
                #confidence_not_equal_zero)
     """
    print 'prediction shape : ', pred.get_shape()
    b = 0
    print 'iou pred shape 0 : ',tf.slice(pred,[0,0,b*5],[-1,-1,4]).get_shape()
    b = 0
        #confidence = tf.concat(0,(confidence,IoU(pred[ : , : , b * 5 : b * 5 + 4], y[ : , b * 5 : b * 5 + 4])))
    #confidence = IoU(pred[ : , : , b * 5 : b * 5 + 4], y[ : , b * 5 : b * 5 + 4])
    confidence = IoU(tf.slice(pred,[0,0,b*5],[-1,-1,4]), y[:,:4])
    b += 1
    test = tf.slice(pred,[0,0,5],[-1,-1,4])
    print 'test shape ', test.get_shape()
    confidence = tf.concat(1,(confidence,IoU(tf.slice(pred,[0,0,b * 5],[-1,-1,4]), y[ : ,:4])))
         #       tf.concat(1,(confidence,IoU(pred[ : , : , b * 5 : b * 5 + 4], y[ : , b * 5 : b * 5 + 4]))))
    """
     if confidence == 0:
            print 'iou xxx ', IoU(pred[ : , : , b * 5 : b * 5 + 4], y[ : , b * 5 : b * 5 + 4]).get_shape()
            confidence = IoU(pred[ : , : , b * 5 : b * 5 + 4], y[ : , b * 5 : b * 5 + 4])
     else:
           # print confidence.dtype
            print  IoU(pred[ : , : , b * 5 : b * 5 + 4], y[ : , b * 5 : b * 5 + 4]).get_shape()
            print 'pred : ',pred[ : , : , b * 5 : b * 5 + 4]
            print 'y :', y[ : , b * 5 : b * 5 + 4]
            print confidence.get_shape
            confidence = tf.concat(1,(confidence,IoU(pred[ : , : , b * 5 : b * 5 + 4], y[ : , b * 5 : b * 5 + 4])))
            print confidence.get_shape()

     """
        #print 'confidence shape : ', confidence.get_shape
    """
        confidence shape = [batch, cell, B]
    """
    confidence = tf.reshape(confidence,shape)

    assert confidence.dtype == tf.float32

    print 'in confidence : ', confidence.get_shape()
    return confidence

def is_responsible(confidence):

    """
    threshold = max of confidence

    so is_res will be a boolean vector, present wheather the cell is responsible for the object

    """

    _, cells, B = list(confidence.get_shape())
    print 'confidence : ', confidence.get_shape()
    max_iou = tf.reduce_max(confidence, 2)
    print 'batch_size :',batch_size
    print 'cells :', cells
    print 'B : ', B
    print 'max_iou : ', max_iou.get_shape()

    for b in xrange(B-1):
        max_iou = tf.concat(1,[max_iou,max_iou])

    print 'max_iou : ', max_iou.get_shape()
    max_iou = tf.reshape(max_iou,[batch_size, int(cells), int(B)])
    is_res = tf.greater_equal(confidence, max_iou)

    #print 'is_res : ', is_res.get_shape()
    #print 'is_res : ', is_res.dtype
    #print 'conf : ', confidence.dtype
    #print 'is_res : ',tf.shape(is_res)
    #print 'confidence : ',tf.shape(confidence)
    #print 'confidence : ',confidence.get_shape()
    assert is_res.dtype == bool

    assert confidence.dtype == tf.float32
    #assert is_res.get_shape() == confidence.get_shape()

    return is_res



def is_appear_in_cell(confidence):

    return tf.greater(tf.reduce_sum(confidence,2),tf.zeros((batch_size,49)))
    #return tf.reduce_all(confidence,2)

"""

training

"""





print 'start training ... '
lcoord = tf.constant(5, dtype = tf.float32)
lnoobj = tf.constant(0.5, dtype = tf.float32)

pred = conv_net(x, weights, biases, 1)
display_step = 1

print 'prediction first : ' , pred.get_shape()
confidence = get_confidence(pred, y, B)
is_res = is_responsible(confidence)
is_appear = is_appear_in_cell(confidence)
not_res = tf.logical_not(is_res)

is_res = tf.cast(is_res, tf.float32)
not_res = tf.cast(is_res, tf.float32)
is_appear = tf.cast(is_appear, tf.float32)

images, objects = load_imdb('plate', cls_name)

images =np.array(images)
loss = None
B = 2
#b = tf.Variable(0)
for b in xrange(B):
#tf.while_loop(b < B, ):

    """
    B = [(SxS) x B]

    x, y => relative to cell
    w, h => relative to image

    pred = [batch, SxS, 5B+C]


    dx = (pred[:,:,b*5+0] - y[:,0]) ** 2
    dy = (pred[:,:,b*5+1] - y[:,1]) ** 2
    dw = (pred[:,:,b*5+2]**0.5 - y[:,2]**0.5) ** 2
    dh = (pred[:,:,b*5+3]**0.5 - y[:,3]**0.5) ** 2
    dc = (pred[:,:,b*5+4] - y[:,4]) ** 2
    """

    pred_x = tf.slice(pred, [0,0,b * 5 + 0], [-1,-1,1])

    print 'slice x : ', tf.slice(y, [0,0], [-1,1]).get_shape()
#    print 'slice x 2 : ', , [-1, S*S, b+1]).
    gt_x = tf.reshape(tf.slice(y, [0,0], [-1,1]), [batch_size, 1, 1])

    pred_y = tf.slice(pred, [0,0,b * 5 + 1], [-1,-1,1])
    gt_y = tf.reshape(tf.slice(y, [0,1], [-1,1]), [batch_size, 1, 1])

    pred_w = tf.slice(pred, [0,0,b * 5 + 2], [-1,-1,1])
    gt_w = tf.reshape(tf.slice(y, [0,2], [-1,1]), [batch_size, 1, 1])

    pred_h = tf.slice(pred, [0,0,b * 5 + 3], [-1,-1,1])
    gt_h = tf.reshape(tf.slice(y, [0,3], [-1,1]), [batch_size, 1, 1])


    pred_c = tf.slice(pred, [0,0,b * 5 + 4], [-1,-1,1])

#    gt_c = tf.ones([-1,S*S,b+1])
    print 'gt_x : ', gt_x.get_shape()
    print 'pred_x : ', pred_x.get_shape()

    dx = tf.pow(tf.sub(pred_x, gt_x), 2)
    dy = tf.pow(tf.sub(pred_y, gt_y), 2)

    dw = tf.pow(tf.sub(tf.pow(pred_w,0.5), tf.pow(gt_w,0.5)), 2)
    dh = tf.pow(tf.sub(tf.pow(pred_h,0.5), tf.pow(gt_h,0.5)), 2)

    dc = tf.pow(tf.sub(pred_c, 1), 2)



    """

    print 'dx predict : ', tf.slice(pred,[0,0,b*5+0],[-1,-1,1]).get_shape()
    print 'dx y :', tf.slice(y,[0,0],[-1,1]).get_shape()
    dx = tf.pow(tf.sub(tf.slice(pred,[0,0,b*5+0],[-1,-1,1]),tf.slice(y,[0,0],[-1,1])),2)
    dy = tf.pow(tf.sub(tf.slice(pred,[0,0,b*5+1],[-1,-1,1]),tf.slice(y,[0,1],[-1,1])),2)
    dw = tf.pow(tf.sub(tf.pow(tf.slice(pred,[0,0,b*5+2],[-1,-1,1]),0.5),tf.pow(tf.slice(y,[0,2],[-1,1]),0.5)),2)
    dh = tf.pow(tf.sub(tf.pow(tf.slice(pred,[0,0,b*5+3],[-1,-1,1]),0.5),tf.pow(tf.slice(y,[0,3],[-1,1]),0.5)),2)

    dc = tf.pow(tf.sub(tf.slice(pred,[0,0,b*5+4],[-1,-1,1]),1),2) #tf.slice(y,[0,4],[-1,1])),2)
    """
    """

    if loss == None:

        loss = lcoord * is_res[:,:,b] * (dx+dy) + \
                lcoord * is_res[:,:,b] * (dw+dh) + \
                is_res[:,:,b] * dc + \
                lnoobj * not_res[:,:,b] * dc
    else:

        loss += lcoord * is_res[:,:,b] * (dx+dy) + \
                lcoord * is_res[:,:,b] * (dw+dh) + \
                is_res[:,:,b] * dc + \
                lnoobj * not_res[:,:,b] * dc

    index = b + 1
    """

    if loss == None:

        #print tf.cast(tf.slice(is_res,[0,0,b],[-1,-1,1]),tf.int32).dtype
        #print tf.add(dx,dy).dtype
        #print lcoord.dtype
        #test1 = tf.cast(tf.slice(is_res,[0,0,b],[-1,-1,1]),tf.float32)
        #test = tf.mul(lcoord, tf.cast(tf.slice(is_res,[0,0,b],[-1,-1,1])))

        loss_coord_xy = tf.mul(tf.mul(lcoord, tf.slice(is_res,[0,0,b],[-1,-1,1])), tf.add(dx,dy))
        loss_coord_wh = tf.mul(tf.mul(lcoord, tf.slice(is_res,[0,0,b],[-1,-1,1])), tf.add(dw,dh))
        loss_is_obj = tf.mul(tf.slice(is_res,[0,0,b],[-1,-1,1]),dc)
        loss_no_obj = tf.mul(tf.slice(not_res,[0,0,b],[-1,-1,1]),dc)

        loss = tf.add(tf.add(loss_coord_xy,loss_coord_wh), tf.add(loss_is_obj,loss_no_obj))

    else:

        loss_coord_xy = tf.mul(tf.mul(lcoord, tf.slice(is_res,[0,0,b],[-1,-1,1])), tf.add(dx,dy))
        loss_coord_wh = tf.mul(tf.mul(lcoord, tf.slice(is_res,[0,0,b],[-1,-1,1])), tf.add(dw,dh))
        loss_is_obj = tf.mul(tf.slice(is_res,[0,0,b],[-1,-1,1]),dc)
        loss_no_obj = tf.mul(tf.slice(not_res,[0,0,b],[-1,-1,1]),dc)

        loss = tf.add(loss, tf.add(tf.add(loss_coord_xy,loss_coord_wh), tf.add(loss_is_obj,loss_no_obj)))

    index = b + 1

    """
loss += is_appear * sum((y[:,:,b:] - pred[:,:,b:]) ** 2)
"""
#print index
#tmp1 =  tf.slice(pred,[0,0,5 * index],[-1,-1,-1])
#print tmp1.get_shape()
#tmp2 = tf.slice(y,[0,5],[-1,-1])
#print tmp2.get_shape(
#print 'is_appear : ', is_appear.dtype
#print 'pred : ',pred.get_shape()
#print 'tmp 1 : ',tf.slice(pred,[0,0,5 * index],[-1,-1,-1]).get_shape()
#tmp = tf.mul(is_appear, tf.pow(tf.reduce_sum(tf.sub(tf.slice(y,[0,4],[-1,-1]), tf.slice(pred,[0,0,5 * index],[-1,-1,-1]))),2))
#print 'tmp shape ', tmp.get_shape()
#print loss.dtype
print 'loss shape : ', loss.get_shape()

"""
reshape loss [batch, cell, bbox] to [batch, bbox], so we can sum over all bbox

"""

gt_cls = tf.pow(tf.sub(tf.slice(y, [0,4], [-1,-1]), tf.slice(pred, [0,0,5 * index], [-1,-1,-1])),2)

print 'gt_cls 1 :', gt_cls.get_shape()

is_appear = tf.reshape(is_appear, [-1, S*S, 1])
gt_cls = tf.mul(is_appear, gt_cls)

print 'gt_cls 2 :', gt_cls.get_shape()
print 'y : ', tf.slice(y,[0,4],[-1,-1]).get_shape()
#loss = tf.add(loss,tf.reduce_sum(tf.mul(is_appear, tf.pow(tf.sub(tf.slice(y,[0,4],[-1,-1]), tf.slice(pred,[0,0,5 * index],[-1,-1,-1])),2))))
print ' b : ', b
#loss = tf.reshape(loss,[int(loss.get_shape()[0]),int(loss.get_shape()[1] * (index))])
#print 'loss shape ', loss.get_shape()
#loss = tf.add(loss, tf.mul(is_appear, tf.pow(tf.reduce_sum(tf.sub(tf.slice(y,[0,4],[-1,-1]), tf.slice(pred,[0,0,5 * index],[-1,-1,-1]))),2)))
#print int(tf.slice(y,[0,4],[-1,-1]).get_shape()[1])
#print tf.slice(y,[0,4],[-1,-1]).get_shape()
assert int(tf.slice(y,[0,4],[-1,-1]).get_shape()[1]) == n_class
loss = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#accuracy =
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#print 'image : ', len(images)
#print 'images : ', images.dtype

#print images.get_shape()








init = tf.initialize_all_variables()
with tf.Session() as sess:


    sess.run(init)
    step = 0

    print 'realy start training ... '
    while step * batch_size < training_iters:
        #print 'image : ', len(images)
        #print 'images : ', images.dtype
        #print len(objects)
        #print images.get_shape()
        #print 'objects:',objects.shape
        #batch_x = tf.slice(images,[step * batch_size,0],[(step+1) * batch_size,-1])
        batch_x = []
        batch_y = []




        start = step * batch_size % len(images)
        end = (step + 1) * batch_size % len(images) # from zero
        if end < start:
            batch_x = images[start : ] + images[:end]
            batch_y = objects[start : ] + objects[:end]
        else:
            batch_x = images[start : end]
            batch_y = objects[start : end]

                #batch_y = objects[step * batch_size : (step+1) * batch_size]

        #batch_x = images[step * batch_size : (step+1) * batch_ize]
        #batch_y = objects[step * batch_size : (step+1) * batch_size]




        print 'batch_y:',batch_y.shape
        print 'batch_x:',batch_x.shape
        #print 'batch_y : ', type(batch_y)
        #print 'batch_x : ', type(batch_x)


        #print 'batch x : ', batch_x.get_shape()
        #print 'batch_y : ', batch_y.get_shape()


        sess.run(optimizer, feed_dict =
                                {
                                 x:batch_x,
                                 y:batch_y
                                })

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
