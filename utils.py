import numpy as np

def IoU(bbox,gt):
    """
    bbox here is a vector , a cell has B bbox ,
    
    """
    x1 = np.maximum(bbox[:,:,0], gt[:,:,0], dtype = float)
    y1 = np.maximum(bbox[:,:,1], gt[:,:,1], dtype = float)
    x2 = np.maximum(bbox[:,:,2], gt[:,:,2], dtype = float)
    y2 = np.maximum(bbox[:,:,3], gt[:,:,3], dtype = float)


    w = x2 - x1 + 1
    h = y2 - y1 + 1
    inter = w * h

    #print inter.shape
    bounding_box = (bbox[:,:,2] - bbox[:,:,0] + 1) * (bbox[:,:,3] - bbox[:,:,1] + 1)
    ground_truth = (gt[:,:,2] - gt[:,:,0] + 1) * (gt[:,:,3] - gt[:,:,1] + 1)
    
    #print bounding_box.shape
    #print ground_truth.shape

    #ground_truth = ground_truth.reshape(len(ground_truth))
    #print ground_truth.shape
    iou = inter / (bounding_box + ground_truth - inter)
    #print 'iou : {}'.format(iou.shape)
    #print 'inter : {}'.format(inter.shape)
    #print 'IoU : {}'. format(iou)

    return iou
