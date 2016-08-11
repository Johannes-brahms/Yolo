import numpy as np
import tensorflow as tf
def IoU(bbox,gt):
    """
    bbox here is a vector , a cell has B bbox ,

    """
    #print bbox[0,:,0]
    #print 'IoU rate : '
    shape = [-1, 1]

    x1 = tf.maximum(tf.cast(bbox[0], tf.float32), tf.reshape(gt[:,0], shape))
    y1 = tf.maximum(tf.cast(bbox[1], tf.float32), tf.reshape(gt[:,1], shape))
    x2 = tf.maximum(tf.cast(bbox[2], tf.float32), tf.reshape(gt[:,2], shape))
    y2 = tf.maximum(tf.cast(bbox[3], tf.float32), tf.reshape(gt[:,3], shape))

    w = tf.add(tf.sub(x2,x1),1)
    h = tf.add(tf.sub(y2,y1),1)

    inter = tf.mul(w,h)

    bounding_box = tf.cast(tf.mul(tf.add(tf.sub(bbox[2], bbox[0]), 1), tf.add(tf.sub(bbox[3],bbox[1]),1)), tf.float32)
    ground_truth = tf.mul(tf.add(tf.sub(gt[:,2], gt[:,0]), 1), tf.add(tf.sub(gt[:,3],gt[:,1]),1))

    #tmp1 = tf.add(bounding_box,tf.reshape(ground_truth,shape))
    #print 'tmp1 : ', tmp1.get_shape()

    #print 'sub : ',tf.sub(tf.add(bounding_box,ground_truth),inter).get_shape()

    iou = tf.div(inter,tf.sub(tf.add(bounding_box,tf.reshape(ground_truth,shape)),inter))

    return iou
