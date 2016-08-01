import numpy as np
import tensorflow as tf
def IoU(bbox,gt):
    """
    bbox here is a vector , a cell has B bbox ,

    """
    #print bbox[0,:,0]
    print 'gt get shape : ', gt.get_shape()
    #print 'IoU rate : '
    shape = [-1, 1]
    x1 = tf.maximum(bbox[:,:,0], tf.reshape(gt[:,0], shape))
    #print x1.get_shape()
    y1 = tf.maximum(bbox[:,:,1], tf.reshape(gt[:,1], shape))
    x2 = tf.maximum(bbox[:,:,2], tf.reshape(gt[:,2], shape))
    y2 = tf.maximum(bbox[:,:,3], tf.reshape(gt[:,3], shape))
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
    #print 'iou : {}'.format(iou.get_shape())
    print 'inter : {}'.format(inter.get_shape())
    print 'bounding_box : ', bounding_box.get_shape()
    print 'ground_truth : ', ground_truth.get_shape()
    tmp1 = tf.add(bounding_box,tf.reshape(ground_truth,shape))
    print 'tmp1 : ', tmp1.get_shape()
    tmp = tf.sub(tmp1,inter)
    print 'sub : ',tf.sub(tf.add(bounding_box,ground_truth),inter).get_shape()

    #print 'IoU : {}'. format(iou)


    iou = tf.div(inter,tmp)

    return iou
