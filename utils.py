import numpy as np
import tensorflow as tf
def log(tensor, string):

    return tf.Print(tensor, [tensor], string)


def IoU(bbox, gt):

    shape = [-1, 1]

    x1 = tf.maximum(tf.cast(bbox[0], tf.float32), tf.reshape(tf.cast(gt[:,0], tf.float32), shape))
    y1 = tf.maximum(tf.cast(bbox[1], tf.float32), tf.reshape(tf.cast(gt[:,1], tf.float32), shape))
    x2 = tf.minimum(tf.cast(bbox[2] + bbox[0], tf.float32), tf.reshape(tf.cast(gt[:,2] + gt[:,0], tf.float32), shape))
    y2 = tf.minimum(tf.cast(bbox[3] + bbox[1], tf.float32), tf.reshape(tf.cast(gt[:,3] + gt[:,1], tf.float32), shape))

    """
    x1_positive_mask = tf.greater(bbox[0], 0)
    y1_positive_mask = tf.greater(bbox[1], 0)
    x2_positive_mask = tf.greater(bbox[2], 0)
    y2_positive_mask = tf.greater(bbox[3], 0)
    
    positive_mask = tf.cast(tf.logical_and(tf.logical_and(x1_positive_mask, y1_positive_mask), tf.logical_and(x2_positive_mask, y2_positive_mask)), tf.float32)
    """

    inter_w = tf.sub(x2,x1)

    inter_h = tf.sub(y2,y1)

    inter = tf.cast(inter_w * inter_h, tf.float32)

    bounding_box = tf.cast(tf.mul(bbox[2],bbox[3]), tf.float32)

    ground_truth = tf.reshape(tf.cast(tf.mul(gt[:,2],gt[:,3]), tf.float32), shape)

    #iou = tf.div(inter,tf.sub(tf.add(bounding_box,tf.reshape(ground_truth,shape)),inter))

    iou = inter / (bounding_box + ground_truth - inter)

    # limit the iou range between 0 and 1
    
    mask_less = tf.cast(tf.logical_not(tf.less(iou, tf.zeros_like(iou))), tf.float32)
    #mask_great = tf.cast(tf.logical_not(tf.greater(iou, tf.ones_like(iou))), tf.float32)
    
    iou = tf.mul(iou, mask_less)
    #iou = tf.mul(iou, positive_mask)
    
    return iou
