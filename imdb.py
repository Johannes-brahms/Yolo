import os
import time
import glob
import lmdb
import gevent
import yolo_pb2
import xml.etree.ElementTree as ET
import numpy as np
import yolo_pb2, base64
from StringIO import StringIO
from PIL import Image
from os.path import join as Path
from gevent.pool import Pool
from skimage import io
from xml.dom import minidom

"""
skimage use RGB to
https://gist.github.com/shelhamer/80667189b218ad570e82#file-readme-md

"""


class ground_truth_object(object):

    def __init__(index, coord):

        self.index = index
        self.coord = coord




g = Pool()

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

def generate_cache(train_list, filename):

    database = lmdb.open(filename, map_size = int(1e12))

    num_imges = 0

    with image_in_db.begin(write = True) as txn:

            for idx, img in enumerate(inputs):

                image = io.imread(img, False)

                assert image.dtype == np.uint8

                data = yolo_pb2.Image()
                data.data = image.tobytes()
                data.height, data.width, data.channels = image.shape
                data.name = img

                for obj in load_annotation_from_xml(img):

                    data.add

                txn.put(str(num_imges), datum.SerializeToString())

                #image = image[:,:,::-1]
                #image = image.transpose((2, 0, 1))

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
    assert len(object) > 0

    datum.object_num = len(object)
    return datum

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


def load_imdb(database):

    env = lmdb.open(database, readonly = True)
    
    objects = []
    images = []

    with env.begin() as txn:

        num = 0

        while True:

            raw = txn.get(str(num))
            
            if raw_datum is None:
                break

            datum = yolo_pb2.Image()

            datum.ParseFromString(raw)

            image = StringIO(base64.b64decode(datum.data))

            image = io.imread(image)

            width = datum.width

            height = datum.height

            channels = datum.channels

            image = image.reshape((height, width, channels))
            
            images.append(image)
            
            assert len(images) == num + 1
            
            for idx in xrange(datum.object_num):

                x = datum.object[idx].x
                y = datum.object[idx].y
                w = datum.object[idx].width
                h = datum.object[idx].height
                cls = datum.object[idx].cls
                objects.append(ground_truth_object(num, [x,y,w,h], cls))

            num += 1

    return images, objects 


        """
        flat_x = np.fromstring(dataum.data, dtype = np.uint8)
        print dataum.object
        print flat_x.shape
        x = flat_x.reshape(dataum.channels, dataum.height, dataum.width)
        print x.shape
        #io.imshow(x.transpose([1,2,0]))
        """
        #io.imshow(image)
        #io.imsave('daw.jpg', image)
        #io.show()

#load_annotation_from_xml('Annotations/xmls/multi/m_1.xml')
#generate_caches('testdb', 'train.txt')
load_imdb('testdb')

"""
threads = []
start = time.time()
image = get_data_from_list('train.txt')
print 'time cost : ', time.time() - start
"""
