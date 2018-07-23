import keras
from keras import Model
from keras import backend as K
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Convolution2D
from keras.layers import merge
from keras.layers import Reshape
from keras.regularizers import l2
from keras.layers import Deconvolution2D
from keras.layers import Input
import matplotlib.pyplot as plt
import numpy as np
import threading
from concurrent.futures import ProcessPoolExecutor
import glob
import random
from PIL import Image
import os
import pickle

# import cv2 as cv

frames_path = './datasets/frames/'
labels_path = './datasets/labels/'
fnames = glob.glob(frames_path + '*.jpg')
lnames = [labels_path + os.path.basename(fn)[:-4] + '.png' for fn in fnames]
img_sz = (480, 360)
print(fnames[0:3])
print(lnames[0:3])


def open_image(fn): return np.array(Image.open(fn).resize(img_sz, Image.NEAREST))


img = Image.open(fnames[0]).resize(img_sz, Image.NEAREST)
# Image._show(img)
imgs = np.stack([open_image(fn) for fn in fnames])
labels = np.stack([open_image(fn) for fn in lnames])
print(imgs.shape, labels.shape)
n = len(labels)
r, c = img.size

# write image and label vlaues into a file
with open('imgs.pickle', 'wb') as pickle_out:
    pickle.dump(imgs, pickle_out)

with open('labels.pickle', 'wb') as pickle_out:
    pickle.dump(labels, pickle_out)

# read image and lable from the file
with open('imgs.pickle', 'rb') as pickle_in:
    imgs = pickle.load(pickle_in)

with open('labels.pickle', 'rb') as pickle_in:
    labels = pickle.load(pickle_in)

print("pickle part completed")
print(type(imgs))


# standardize
# imgs-=0.4
# imgs/=0.3

class BatchIndices(object):
    def __init__(self, n, bs, shuffle=False):
        self.n, self.bs, self.shuffle = n, bs, shuffle
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.idxs = (np.random.permutation(self.n)
                     if self.shuffle else np.arange(0, self.n))
        self.curr = 0

    def __next__(self):
        with self.lock:
            if self.curr >= self.n: self.reset()
            ni = min(self.bs, self.n - self.curr)
            res = self.idxs[self.curr:self.curr + ni]
            self.curr += ni
            return res


class segm_generator(object):
    def __init__(self, x, y, bs=64, out_sz=(224, 224), train=True):
        self.x, self.y, self.bs, self.train = x, y, bs, train
        self.n, self.ri, self.ci, _ = x.shape
        self.idx_gen = BatchIndices(self.n, bs, train)
        self.ro, self.co = out_sz
        self.ych = self.y.shape[-1] if len(y.shape) == 4 else 1

    def get_slice(self, i, o):
        start = random.randint(0, i - o) if self.train else (i - o)
        return slice(start, start + o)

    def get_item(self, idx):
        slice_r = self.get_slice(self.ri, self.ro)
        slice_c = self.get_slice(self.ci, self.co)
        x = self.x[idx, slice_r, slice_c]
        y = self.y[idx, slice_r, slice_c]
        if self.train and (random.random() > 0.5):
            y = y[:, ::-1]
            x = x[:, ::-1]
        return x, y

    def __next__(self):
        idxs = next(self.idx_gen)
        items = (self.get_item(idx) for idx in idxs)
        xs, ys = zip(*items)
        return np.stack(xs), np.stack(ys).reshape(len(ys), -1, self.ych)


##converting labels

# def parse_code(l):
#    a,b = l.strip('\n').split("\t")
#    return tuple(int(o) for o in a.split(' ')), b
#
# label_codes,label_names = zip(*[
#    parse_code(l) for l in open("./label_colors.txt")])

label_codes, label_names = [48.8, 10.7, 5.7], ["head"]

code2id = {v: k for k, v in enumerate(label_codes)}

failed_code = len(label_codes) + 1

label_codes.append((0, 0, 0))
label_names.append('unk')


def conv_one_label(i):
    res = np.zeros((r, c), 'uint8')
    for j in range(r):
        for k in range(c):
            try:
                res[j, k] = code2id[tuple(labels[i, j, k])]
            except:
                res[j, k] = failed_code
    return res


def conv_all_labels():
    ex = ProcessPoolExecutor(8)
    return np.stack(ex.map(conv_one_label, range(n)))


labels_int = conv_all_labels()

np.count_nonzero(labels_int == failed_code)

labels_int[labels_int == failed_code] = 0

with open("labels_int.pickle", "wb") as pickle_out:
    pickle.dump(labels_int, pickle_out)

with open("labels_int.pickle", "rb") as pickle_in:
    labels_int = pickle.load(pickle_in)

# show result
sg = segm_generator(imgs, labels, 4, train=True)
b_img, b_label = next(sg)
plt.imshow(b_img[0]);

plt.imshow(b_label[0].reshape(224, 224));
print(b_label[0])

# preparing test set
"""
fn_test = set(o.strip() for o in open('test.txt', 'r'))
is_test = np.array([o.split('/')[-1] in fn_test for o in fnames])
trn = imgs[is_test == False]
trn_labels = labels_int[is_test == False]
test = imgs[is_test]
test_labels = labels_int[is_test]
print(trn.shape, test_labels.shape)
rnd_trn = len(trn_labels)
rnd_test = len(test_labels)
"""

# tiramisu network

def relu(x): return Activation('relu')(x)


def dropout(x, p): return Dropout(p)(x) if p else x


def bn(x): return BatchNormalization(mode=2, axis=-1)(x)


def relu_bn(x): return relu(bn(x))


def concat(xs): return merge(xs, mode='concat', concat_axis=-1)


def conv(x, nf, sz, wd, p, stride=1):
    x = Convolution2D(nf, sz, sz, init='he_uniform', border_mode='same',
                      subsample=(stride, stride), W_regularizer=l2(wd))(x)
    return dropout(x, p)


def conv_relu_bn(x, nf, sz=3, wd=0, p=0, stride=1):
    return conv(relu_bn(x), nf, sz, wd=wd, p=p, stride=stride)


def dense_block(n, x, growth_rate, p, wd):
    added = []
    for i in range(n):
        b = conv_relu_bn(x, growth_rate, p=p, wd=wd)
        x = concat([x, b])
        added.append(b)
    return x, added


# Downsampling transition
def transition_dn(x, p, wd):
    #     x = conv_relu_bn(x, x.get_shape().as_list()[-1], sz=1, p=p, wd=wd)
    #     return MaxPooling2D(strides=(2, 2))(x)
    return conv_relu_bn(x, x.get_shape().as_list()[-1], sz=1, p=p, wd=wd, stride=2)


# making downword path
def down_path(x, nb_layers, growth_rate, p, wd):
    skips = []
    for i, n in enumerate(nb_layers):
        x, added = dense_block(n, x, growth_rate, p, wd)
        skips.append(x)
        x = transition_dn(x, p=p, wd=wd)
    return skips, added


# Up sampling
def transition_up(added, wd=0):
    x = concat(added)
    _, r, c, ch = x.get_shape().as_list()
    return Deconvolution2D(ch, 3, 3, (None, r * 2, c * 2, ch), init='he_uniform',
                           border_mode='same', subsample=(2, 2), W_regularizer=l2(wd))(x)


#     x = UpSampling2D()(x)
#     return conv(x, ch, 2, wd, 0)

# Making upward path
def up_path(added, skips, nb_layers, growth_rate, p, wd):
    for i, n in enumerate(nb_layers):
        x = transition_up(added, wd)
        x = concat([x, skips[i]])
        x, added = dense_block(n, x, growth_rate, p, wd)
    return x


#tiramisu model

def reverse(a): return list(reversed(a))


def create_tiramisu(nb_classes, img_input, nb_dense_block=6,
                    growth_rate=16, nb_filter=48, nb_layers_per_block=5, p=None, wd=0):
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)
    else:
        nb_layers = [nb_layers_per_block] * nb_dense_block

    x = conv(img_input, nb_filter, 3, wd, 0)
    skips, added = down_path(x, nb_layers, growth_rate, p, wd)
    x = up_path(added, reverse(skips[:-1]), reverse(nb_layers[:-1]), growth_rate, p, wd)

    x = conv(x, nb_classes, 1, wd, 0)
    _, r, c, f = x.get_shape().as_list()
    x = Reshape((-1, nb_classes))(x)
    return Activation('softmax')(x)

#Training

input_shape = (224,224,3)
img_input = Input(shape=input_shape)
x = create_tiramisu(12, img_input, nb_layers_per_block=[4,5,7,10,12,15], p=0.2, wd=1e-4)
model = Model(img_input, x)
gen = segm_generator(imgs, labels, 3, train=True)
#gen_test = segm_generator(test, test_labels, 3, train=False)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop(1e-3), metrics=["accuracy"])
model.optimizer=keras.optimizers.RMSprop(1e-3, decay=1-0.99995)
model.optimizer=keras.optimizers.RMSprop(1e-3)
K.set_value(model.optimizer.lr, 1e-3)
model.fit_generator(gen, len(labels), 100, verbose=2)

#validation_data=gen_test, nb_val_samples=rnd_test



