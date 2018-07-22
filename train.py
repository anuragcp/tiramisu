import keras
import matplotlib.pyplot as plt
import numpy as np
import threading
from concurrent.futures import ProcessPoolExecutor
import glob
from PIL import Image
import os
import pickle

frames_path = './datasets/frames/'
labels_path = './datasets/labels/'
fnames = glob.glob(frames_path+'*.jpg')
lnames = [labels_path+os.path.basename(fn)[:-4]+'.png' for fn in fnames]
img_sz = (480,360)
print(fnames[0:3])
print(lnames[0:3])
def open_image(fn): return np.array(Image.open(fn).resize(img_sz, Image.NEAREST))
#img = Image.open(fnames[0]).resize(img_sz, Image.NEAREST)
imgs = np.stack([open_image(fn) for fn in fnames])
labels = np.stack([open_image(fn) for fn in lnames])
print(imgs.shape,labels.shape)

# write image and label vlaues into a file
with open('imgs.bc', 'w') as file:
	file.write(np.ndarray(imgs))

with open('labels.bc', 'w') as file:
        file.write(np.ndarray(labels))

"""
imgs = load_array(PATH+'results/imgs.bc')
labels = load_array(PATH+'results/labels.bc')

#standardize
imgs-=0.4
imgs/=0.3


class BatchIndices(object):
    def __init__(self, n, bs, shuffle=False):
        self.n,self.bs,self.shuffle = n,bs,shuffle
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.idxs = (np.random.permutation(self.n)
                     if self.shuffle else np.arange(0, self.n))
        self.curr = 0

    def __next__(self):
        with self.lock:
            if self.curr >= self.n: self.reset()
            ni = min(self.bs, self.n-self.curr)
            res = self.idxs[self.curr:self.curr+ni]
            self.curr += ni
            return res

class segm_generator(object):
    def __init__(self, x, y, bs=64, out_sz=(224,224), train=True):
        self.x, self.y, self.bs, self.train = x,y,bs,train
        self.n, self.ri, self.ci, _ = x.shape
        self.idx_gen = BatchIndices(self.n, bs, train)
        self.ro, self.co = out_sz
        self.ych = self.y.shape[-1] if len(y.shape)==4 else 1

    def get_slice(self, i,o):
        start = random.randint(0, i-o) if self.train else (i-o)
        return slice(start, start+o)

    def get_item(self, idx):
        slice_r = self.get_slice(self.ri, self.ro)
        slice_c = self.get_slice(self.ci, self.co)
        x = self.x[idx, slice_r, slice_c]
        y = self.y[idx, slice_r, slice_c]
        if self.train and (random.random()>0.5):
            y = y[:,::-1]
            x = x[:,::-1]
        return x, y

    def __next__(self):
        idxs = next(self.idx_gen)
        items = (self.get_item(idx) for idx in idxs)
        xs,ys = zip(*items)
        return np.stack(xs), np.stack(ys).reshape(len(ys), -1, self.ych)

##converting labels

def parse_code(l):
    a,b = l.strip().split("\t")
    return tuple(int(o) for o in a.split(' ')), b

label_codes,label_names = zip(*[
    parse_code(l) for l in open("./label_colors.txt")])

label_codes,label_names = list(label_codes),list(label_names)

code2id = {v:k for k,v in enumerate(label_codes)}


failed_code = len(label_codes)+1

label_codes.append((0,0,0))
label_names.append('unk')

def conv_one_label(i):
    res = np.zeros((r,c), 'uint8')
    for j in range(r):
        for k in range(c):
            try: res[j,k] = code2id[tuple(labels[i,j,k])]
            except: res[j,k] = failed_code
    return res

def conv_all_labels():
    ex = ProcessPoolExecutor(8)
    return np.stack(ex.map(conv_one_label, range(n)))

%time labels_int =conv_all_labels()

np.count_nonzero(labels_int==failed_code)

labels_int[labels_int==failed_code]=0

save_array(PATH+'results/labels_int.bc', labels_int)

labels_int = load_array(PATH+'results/labels_int.bc')

#show result
sg = segm_generator(imgs, labels, 4, train=True)
b_img, b_label = next(sg)
plt.imshow(b_img[0]*0.3+0.4);

plt.imshow(b_label[0].reshape(224,224,3));
"""
