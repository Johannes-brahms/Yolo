import numpy as np
import tensorflow as tf
def IoU(bbox,gt):
    """
    bbox here is a vector , a cell has B bbox ,
    
    """
    #print bbox[0,:,0]
    #print gt[:,0]
    #print 'IoU rate : '
    x1 = tf.maximum(bbox[:,:,0], gt[:,0])
    #print x1.get_shape()
    y1 = tf.maximum(bbox[:,:,1], gt[:,1])
    x2 = tf.maximum(bbox[:,:,2], gt[:,2])
    y2 = tf.maximum(bbox[:,:,3], gt[:,3])
    #y2 = np.maximum(bbox[:,:,3], gt[:,3], dtype = float)


    w = tf.add(tf.sub(x2,x1),1)
    h = tf.add(tf.sub(y2,y1),1)

    inter = tf.mul(w,h)



    #print inter.shape
    bounding_box = tf.mul(tf.add(tf.sub(bbox[:,:,2], bbox[:,:,0]), 1), tf.add(tf.sub(bbox[:,:,3],bbox[:,:,1]),1))

    #ground_truth = (gt[:,:,2] - gt[:,:,0] + 1) * (gt[:,:,3] - gt[:,:,1] + 1)
    
    ground_truth = tf.mul(tf.add(tf.sub(gt[:,2], gt[:,0]), 1), tf.add(tf.sub(gt[:,3],gt[:,1]),1))
    
    #print 'gt: xxx : ', ground_truth.get_shape()
    #print bounding_box.shape
    #print ground_truth.shape

    #ground_truth = ground_truth.reshape(len(ground_truth))
    #print ground_truth.shape
    iou = inter / tf.sub(tf.add(bounding_box,ground_truth),inter)
    #print 'iou : {}'.format(iou.get_shape())
    #print 'inter : {}'.format(inter.shape)
    #print 'IoU : {}'. format(iou)

    return iou
