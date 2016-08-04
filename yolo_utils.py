import tensorflow as tf
def cell_locate(shape,bbox, S):

    height, width = shape

    x, y, w, h = bbox

    cell_w = width / S
    cell_h = height / S

    center_x = (x + w) * 0.5
    center_y = (y + h) * 0.5

    cell_coord_x = center_x / cell_w
    cell_coord_y = center_y / cell_h


    cell_num = cell_coord_y * S + cell_coord_x
    

    # 0 , 0 at left up , offset range between only in range ( 0 , 1)
    offset_x = float(center_x - cell_coord_x * cell_w) / cell_w
    offset_y = float(center_y - cell_coord_y * cell_h) / cell_h

    assert offset_x < 1 and offset_y < 1 and offset_x > 0 and offset_y > 0

    return cell_num, [offset_x, offset_y, float(w / width), float(h / height)]




def cell_locate(size, bbox, S):

    """ 
    locate the center of ground truth in which grid cell

    """


    x = tf.slice(bbox, [0,0], [-1,1])
    y = tf.slice(bbox, [0,1], [-1,1])
    w = tf.slice(bbox, [0,2], [-1,1])
    h = tf.slice(bbox, [0,3], [-1,1])


    height, width = size

    cell_w = width / S
    cell_h = height / S

    center_x = tf.mul(tf.add(x, w), 0.5)
    center_y = tf.mul(tf.add(y, h), 0.5)


    cell_coord_x = tf.div(center_x, cell_w)
    cell_coord_y = tf.div(center_y, cell_h)


    cell_num = tf.add(tf.mul(cell_coord_y, S), cell_coord_x)


    #offset_x = tf.div(tf.sub(center_x, tf.mul(cell_coord_x, cell_w)), cell_w)
    #offset_y = tf.div(tf.sub(center_y, tf.mul(cell_coord_y, cell_h)), cell_h)

    #print 'cell num : ', cell_num.get_shape()
    #print 'offset x : ', offset_x.get_shape()

    return cell_num#, offset_x, offset_y
        



def convert_to_one(bbox, width, height, S):

    x, y, w, h = bbox

    center_x = tf.mul(tf.add(x, w), 0.5)
    center_y = tf.mul(tf.add(y, h), 0.5)

    w = tf.div(w, width)
    h = tf.div(h, height)

    cell_w = width / S
    cell_h = height / S

    cell_coord_x = tf.div(center_x, cell_w)
    cell_coord_y = tf.div(center_y, cell_h)

    offset_x = tf.div(tf.sub(center_x, tf.mul(cell_coord_x, cell_w)), cell_w)
    offset_y = tf.div(tf.sub(center_y, tf.mul(cell_coord_y, cell_h)), cell_h)

    assert offset_x.dtype == tf.float32 and \
            offset_y.dtype == tf.float32 and \
            w.dtype == tf.float32 and \
            h.dtype == tf.float32

    bbox = [offset_x, offset_y, w, h]

    return bbox