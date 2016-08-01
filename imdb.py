import os
import time
import glob
import lmdb
import gevent
import yolo_pb2
import xml.etree.ElementTree as ET
import numpy as np
import yolo_pb2, base64
import tensorflow as tf
from StringIO import StringIO
from PIL import Image
from os.path import join as Path
from gevent.pool import Pool
from skimage import io
from skimage.transform import resize
from xml.dom import minidom
from PIL import Image
import gc
"""
skimage use RGB to
https://gist.github.com/shelhamer/80667189b218ad570e82#file-readme-md

"""


def get_data_from_list(train_list):

    train_list = open(train_list, 'r')

    image_queue = []

    for image in train_list:

        image = image.strip('\n')
        print 'image :', image
        g.spawn(image_queue.append(io.imread(os.path.join('Images',image + '.jpg'), True)))
        g.join()

        print 'image :', image

    return image_queue

def merge_roidbs(filename, datum):

    path = Path('Annotations', 'xmls', filename + '.xml')

    xml = ET.parse(path)

    objects = xml.findall('object')

    for ix, obj in enumerate(objects):

        bbox = obj.find('bndbox')

        # Make pixel indexes 0-based

        xmin = float(bbox.find('xmin').text) # - 1
        ymin = float(bbox.find('ymin').text) # - 1
        xmax = float(bbox.find('xmax').text) # - 1
        ymax = float(bbox.find('ymax').text) # - 1

        cls = obj.find('name').text

        #print 'class : {}, xmin : {}, ymix : {}, xmax : {}, ymax : {}'.format(cls, xmin, ymin, xmax, ymax)

        #obj.append((cls, xmin, ymin, xmax, ymax))

        o = datum.object.add()
        o.x = int(xmin)
        o.y = int(ymin)
        o.width = int(xmax - xmin)
        o.height = int(ymax - ymin)
        o.cls = cls
    assert len(objects) > 0

    datum.object_num = len(objects)
    return datum
def get_index_by_name(cls, cls_name):

    cls_num = len(cls_name)
    index = cls_name.index(cls)
    gt = np.zeros(cls_num)

    gt[index] = 1
    #print gt
    #print gt.shape
    return gt


def generate_caches(database, lists):

    env = lmdb.open(database, map_size = int(1e12))

    image_list = open(lists, 'r')

    with env.begin(write = True) as txn:

        num = 0

        for image_name in image_list:

            image_name = image_name.strip('\n')

            with open(Path('Images', image_name + '.jpg'), 'r') as image:

                data = base64.b64encode(image.read())

                image = io.imread(Path('Images', image_name + '.jpg'), False)

                datum = yolo_pb2.Image()

                datum.data = data
                datum.channels = image.shape[2]
                datum.height = image.shape[0]
                datum.width = image.shape[1]

                datum = merge_roidbs(image_name, datum)
                txn.put(str(num), datum.SerializeToString())
                print 'Images {}'.format(num)
            num += 1


def test_proto():
    import yolo_pb2

    env = lmdb.open('test_db', map_size = int(1e12))

    with env.begin(write = True) as txn:
        image = io.imread('ddd.JPG', False).transpose([2,0,1])
        datum = yolo_pb2.Image()



        datum.data = image.tobytes()
        datum.channels = image.shape[0]
        datum.height = image.shape[1]
        datum.width = image.shape[2]

        Object = datum.object.add()
        Object.x = 0
        Object.y = 0
        Object.width = 0
        Object.height = 0
        Object.cls = 'plate'

        Object = datum.object.add()
        Object.x = 4
        Object.y = 4
        Object.width = 7
        Object.height = 7
        Object.cls = 'cat'
        txn.put('test', datum.SerializeToString())


def load_imdb(database, cls_name):
    start = time.time()
    env = lmdb.open(database, readonly = True)
    print 'loading database .... '
    objects = None
    images = []

    with env.begin() as txn:

        num = 0

        while True:

            raw = txn.get(str(num))

            if raw is None:
                break

            datum = yolo_pb2.Image()

            datum.ParseFromString(raw)

            image = StringIO(base64.b64decode(datum.data))

            image = io.imread(image)

            original_width = datum.width

            original_height = datum.height

            channels = datum.channels

            image = image.reshape((original_height, original_width, channels))

            image = resize(image, (448,448))

            width = 448

            w_ratio = float(original_width) / width

            height = 448

            h_ratio = float(original_height) / height

            #io.imshow(image)
            #io.show()
            for idx in xrange(datum.object_num):

                x = float(datum.object[idx].x) / w_ratio / width
                y = float(datum.object[idx].y) / h_ratio / height
                w = float(datum.object[idx].width) / w_ratio / width
                h = float(datum.object[idx].height) / h_ratio / height

                cls = datum.object[idx].cls
                gt_cls = get_index_by_name(cls, cls_name)

                if objects == None:
                    objects = np.hstack((np.array([x,y,w,h]),gt_cls))
                else:
                    objects = np.vstack((objects, np.hstack((np.array([x,y,w,h]),gt_cls))))

                #objects.append([x,y,w,h,gt_cls])
                images.append(image)


                print 'load Images : {}'.format(num)

            if num % 1000 == 0:
                gc.collect()

            num += 1

    #print len(objects)
    #print len(images)

    assert len(objects) == len(images)

    print 'consume : ', time.time() - start
    return images, objects



#load_annotation_from_xml('Annotations/xmls/multi/m_1.xml')
#generate_caches('plate', 'train.txt')
load_imdb('plate',['plate'])
