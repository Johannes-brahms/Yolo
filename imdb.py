import os, time
import lmdb, cv2
import xml.etree.ElementTree as ET
import numpy as np
import yolo_pb2, base64
import tensorflow as tf
from StringIO import StringIO
from PIL import Image
from os.path import join as Path
from skimage import io
from yolo_utils import cell_locate

"""
skimage use RGB to
https://gist.github.com/shelhamer/80667189b218ad570e82#file-readme-md

"""

def merge_roidbs(filename, datum, ratio, original):

    path = Path('Annotations', 'xml', filename + '.xml')

    xml = ET.parse(path)

    w_ratio, h_ratio = ratio

    print 'Filename : ', filename

    objects = xml.findall('object')

    for ix, obj in enumerate(objects):

        bbox = obj.find('bndbox')

        # Make pixel indexes 0-based

        xmin = float(bbox.find('xmin').text)  #/ datum.width# - 1
        ymin = float(bbox.find('ymin').text)  #/ datum.height# - 1
        xmax = float(bbox.find('xmax').text)   #/ datum.width# - 1
        ymax = float(bbox.find('ymax').text)   #/ datum.height# - 1

        print 'original w : {}, h : {}'.format(original[1], original[0]) 
        print 'xmax : {}, ymax : {}'.format(xmax, ymax)

        assert xmax <= original[1] and ymax <= original[0]

        xmin /= w_ratio #/ datum.width# - 1
        ymin /= h_ratio #/ datum.height# - 1
        xmax /= w_ratio #/ datum.width# - 1
        ymax /= h_ratio 

        cls = obj.find('name').text


        o = datum.object.add()
        o.x = int(xmin)
        o.y = int(ymin)

        o.width = int(xmax - xmin)
        o.height = int(ymax - ymin)

        try :

            assert o.x + o.width <= 448 and o.y + o.height <= 448

        except:
            print '=================='
            print 'filename : ', filename
            """
            print 'original x min : {} , y min : {}, x max : {}, y max : {} '.format(float(bbox.find('xmin').text),float(bbox.find('ymin').text)
            , original_width, original_height)


            print 'resize x min : {} , y min : {}, x max : {}, y max : {} '.format(xmin, ymin, new_width, new_height)
            print '=================='
            """
            raise
          

        o.cls = cls

    assert len(objects) > 0

    datum.object_num = len(objects)

    return datum
def get_index_by_name(cls, cls_name):
    #print 'class ', cls
    cls_num = len(cls_name)
    index = cls_name.index(cls)
    gt = np.zeros(cls_num)
    gt[index] = 1
    return gt
def generate_caches_with_raw(database, lists):

    with lmdb.open(database, map_size = int(1e10)) as env :

        image_list = open(lists, 'r')
        start = time.time()
        num = 0

        for image_name in image_list:

            with env.begin(write = True) as txn :

                image_name = image_name.strip('\n')
                image = io.imread(Path('Images', image_name + '.jpg'), False)
                print 'image shape : ', image.shape
                w_ratio = float(image.shape[1]) / 448
                h_ratio = float(image.shape[0]) / 448

                original = image.shape

                # resize image to 448 x 448
                image = cv2.resize(image, (448, 448))

                # parse to datum format
                datum = yolo_pb2.Image_raw()
                datum.data = image.tobytes()
                datum.channels = image.shape[2]
                datum.height = image.shape[0]
                datum.width = image.shape[1]
                datum.filename = image_name.encode()

                datum = merge_roidbs(image_name, datum, [w_ratio, h_ratio], original)
                # write to database
                txn.put(str(num), datum.SerializeToString())

                num += 1

                print 'load images : {}'.format(num)

    print 'time : ', time.time() - start


def load_imdb_from_raw(database, cls_name):

    start = time.time()
    env = lmdb.open(database, readonly = True)
    print '\n\n[*] loading database .... '
    objects = None
    images = []
    filename = []
    num = 0
    with env.begin() as txn:
        while True:
            raw = txn.get(str(num))
            if type(raw) is not str:
                break

            datum = yolo_pb2.Image_raw()
            datum.ParseFromString(raw)

            image = np.fromstring(datum.data, dtype=np.uint8)          
            image = image.reshape((datum.height,  datum.width,  datum.channels))

            for idx in xrange(datum.object_num):

                x = datum.object[idx].x         #/ w_ratio / width
                y = datum.object[idx].y         #/ h_ratio / height
                w = datum.object[idx].width     #/ w_ratio / width
                h = datum.object[idx].height    #/ h_ratio / height


                cls = datum.object[idx].cls
                gt_cls = get_index_by_name(cls, cls_name)

                if type(objects) != np.ndarray:
                    objects = np.hstack((np.array([x, y, w, h]), gt_cls))#, object_center))
                else:
                    objects = np.vstack((objects, np.hstack((np.array([x, y, w, h]), gt_cls))))#, object_center))))

                images.append(image.flatten())
                filename.append(datum.filename)
                # print type(image.flatten())
                # print 'load Images : {}'.format(num)

            num += 1
            # print 'load Images : {}'.format(num)

    assert len(objects) == len(images)
    print '[*] image loading is done ...'
    print '[*] consume :', time.time() - start
    #images = np.array(images, dtype = np.int32)

    #print images.dtype
    #print images.shape
    return images, objects, filename


def generate_caches_with_jpg(database, lists):

    with lmdb.open(database, map_size = int(1e12)) as env :

        image_list = open(lists, 'r')

        with env.begin(write = True) as txn:

            num = 0

            for image_name in image_list:

                image_name = image_name.strip('\n')

                with open(Path('Images', image_name + '.jpg'), 'r') as image:

                    data = base64.b64encode(image.read())

                    image = io.imread(Path('Images', image_name + '.jpg'), False)

                    w_ratio = float(image.shape[1]) / 448
                    h_ratio = float(image.shape[0]) / 448

                    image = cv2.resize(image,(448,448))
                    datum = yolo_pb2.Image_jpg()
                    datum.data = data
                    datum.channels = image.shape[2]
                    datum.height = image.shape[0]
                    datum.width = image.shape[1]

                    datum = merge_roidbs(image_name, datum, [w_ratio, h_ratio])
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
"""
def load_imdb_from_jpg(database, cls_name, start, end):
    start = time.time()

    with lmdb.open(database, readonly = True) as env:
        print 'loading database .... '
        objects = None
        images = []

        with env.begin() as txn:
            num = 0
            while True:
                raw = txn.get(str(num))
                if raw is None:
                    break
                datum = yolo_pb2.Image_jpg()
                datum.ParseFromString(raw)
                image = StringIO(base64.b64decode(datum.data))
                image = io.imread(image)

                original_width = datum.width
                original_height = datum.height
                channels = datum.channels

                image = image.reshape((original_height, original_width, channels))
                image = cv2.resize(image, (448,448))

                width = 448
                height = 448
                w_ratio = float(original_width) / width
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
                        #images = np.reshape(image,(1,448,448,3))
                        #images = image.flatten()
                    else:
                        objects = np.vstack((objects, np.hstack((np.array([x,y,w,h]),gt_cls))))
                        #images = np.vstack((images,np.reshape(image,(1,448,448,3))))
                        #images = np.hstack(image.flatten())
                        #print images.shape

                    images.append(image)



                    print 'load Images : {}'.format(num)

                #if num == 100:
                    #break
                if num % 1000 == 0:
                    gc.collect()
                num += 1
        assert len(objects) == len(images)
        print 'consume : ', time.time() - start
    return images, objects
"""

"""
def parallel(database, cores, cls_name):
    env = lmdb.open(database, readonly = True)
    with env.begin() as txn:
        length = txn.stat()['entries']
        print 'length : ', length

        batch = length / cores

        threads = []
        start = time.time()
        for core in xrange(cores):

            begin = batch * core

            if core == cores:
                end = length - 1
            else:
                end = batch * (core + 1)

            threads.append(gevent.spawn(load_imdb_from_raw, database, cls_name, begin, end))
            #thread.start_new_thread( load_imdb_from_raw, (database, cls_name, begin, end) )

        start = time.time()
        gevent.joinall(threads)
        print 'consume : ', time.time() - start


"""

if __name__ == '__main__':

    generate_caches_with_raw('char', 'Images/train.txt')
    #generate_caches_with_raw('plate_db', '5000.txt')
#load_imdb_from_jpg('plate',['plate'])
#load_imdb_from_raw('5000_raw',['plate'])
