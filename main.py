import matplotlib.pyplot as plt
import os
import numpy as np
import gzip
import heapq


# -----------------------------------------------------------------------------------------------------------------------
def nflatten(ar):
    return ar.reshape(ar.shape[0], ar.shape[1] * ar.shape[2])


# -----------------------------------------------------------------------------------------------------------------------
def histogram(ar):
    ar = nflatten(ar)
    ret = np.zeros((ar.shape[0], 256))
    idx = 0
    for i in ar:
        for j in i:
            ret[idx][int(j)] += 1
        idx += 1
    return ret


# -----------------------------------------------------------------------------------------------------------------------
def get_pixel(ar, x, y, size, t):
    Max = 0
    Min = 1000
    Avg = 0
    for i in range(x, x + size):
        for j in range(y, y + size):
            Max = max(Max, ar[i][j])
            Min = min(Min, ar[i][j])
            Avg += ar[i][j]
    if (t == 'max'):
        return Max
    if (t == 'min'):
        return Min
    return Avg / (size * size)


# -----------------------------------------------------------------------------------------------------------------------
def down_sample(ar, size, t):
    ret = np.zeros((ar.shape[0], size, size))
    step = 28 // size
    for pic in range(ret.shape[0]):
        for i in range(0, 28, step):
            for j in range(0, 28, step):
                ret[pic][i // step][j // step] = get_pixel(ar[pic], i, j, step, t)
    return ret


# -----------------------------------------------------------------------------------------------------------------------
# test
# a = np.arange(32).reshape(2,4,4)
# print(a)
# print(down_sample(a,2,'avg'))
# -----------------------------------------------------------------------------------------------------------------------
def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)
    with gzip.open(labels_path, 'rb') as lbpath:
        lbpath.read(8)
        buffer = lbpath.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)
    with gzip.open(images_path, 'rb') as imgpath:
        imgpath.read(16)
        buffer = imgpath.read()
        images = np.frombuffer(buffer, dtype=np.uint8).reshape(len(labels), 28,
                                                               28).astype(np.float64)
    return images, labels


# -----------------------------------------------------------------------------------------------------------------------
X_train, y_train = load_mnist('data/', kind='train')
X_test, y_test = load_mnist('data/', kind='t10k')
y_ans = np.ones(y_test.shape) * -1


# X_train = X_train[:6000]
# X_test = X_test[:20]


# -----------------------------------------------------------------------------------------------------------------------
def TEST_ERROR_RATE():
    Tru = 0
    for i in range(y_test.shape[0]):
        if (y_test[i] == y_ans[i]):
            Tru += 1
    return Tru


# -----------------------------------------------------------------------------------------------------------------------
def init(type, size=14, t='avg'):
    global X_train, X_test
    print(type)
    if (type == 'histogram'):
        X_train = histogram(X_train)
        X_test = histogram(X_test)
    if (type == 'downsampling'):
        #X_train = down_sample(X_train, size, 'avg')
        #X_test = down_sample(X_test, size, 'avg')
        X_train = nflatten(X_train)
        X_test = nflatten(X_test)
    print('done')


# -----------------------------------------------------------------------------------------------------------------------
def distance(a, b):
    ret = 0
    for i in range(a.shape[0]):
        ret += abs(a[i] - b[i]) ** 2
    return ret ** (0.5)


# -----------------------------------------------------------------------------------------------------------------------
def KNN(input ,X_train, y_train, k=50):
    size=X_train.shape[0]
    heap = []
    for i in range(k):
        heapq.heappush(heap, (-distance(input, X_train[i]), y_train[i]))
    for i in range(k, size):
        heapq.heappushpop(heap, (-distance(input, X_train[i]), y_train[i]))
    sl = np.zeros((10))
    for i in heap:
        sl[i[1]] += 1
    return np.argmax(sl)

# -----------------------------------------------------------------------------------------------------------------------
def AVG(input ,X_train, y_train):
    avg_train = np.zeros((10,X_train.shape[1]))
    count= np.zeros((10))
    for i in range(X_train.shape[0]):
        avg_train[int(y_train[i])]+=X_train[i]
        count[y_train[i]]+=1
    for i in range(10):
        avg_train[i]/=count[i]
    avg_lable=[0,1,2,3,4,5,6,7,8,9]

    return KNN(input,avg_train,avg_lable,1)

# -----------------------------------------------------------------------------------------------------------------------
init('downsampling', 14)
print(KNN(X_test[0],X_train,y_train), y_test[0])
print(AVG(X_test[0],X_train,y_train), y_test[0])

