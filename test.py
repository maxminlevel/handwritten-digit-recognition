import matplotlib.pyplot as plt
import os
import numpy as np
import gzip
import math

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)
    
    with gzip.open(labels_path, 'rb') as lbpath:
        lbpath.read(8)
        buffer = lbpath.read()
        labels = np.frombuffer(buffer, dtype = np.uint8) # doc 1 lan 8bit = 1byte
    with gzip.open(images_path, 'rb') as imgpath:
        imgpath.read(16)
        buffer = imgpath.read()
        images = np.frombuffer(buffer, dtype = np.uint8).reshape(len(labels), 28,28).astype(np.float64)
    
    return images, labels

def process(X_train):
    Z_train = np.zeros((60000,784))
    for i in range(len(X_train)):
        Z_train[i] = (X_train[i].flatten())
    return Z_train

def calchis(Z_train):    
    his = np.zeros((60000,256))
    for i in range(len(Z_train)):
        for j in range(len(Z_train[i])):
            his[i][int(Z_train[i][j])]+=1
    
    return his

def downsampling(a,d,pre):
    ans = np.zeros((60000,int(28//d*28//d))) #tao mang 1 chieu ( 28/d )^2
    res = np.zeros((60000,int(28//d),int(28//d)))
    for k in range(len(a)):
        for i in range(d):
            for j in range(d):
                summ, maxx, minn = 0, 256, 0
                for u in range(i * 28//d, (i+1) * 28//d, 1):
                        for v in range(j * 28//d, (j+1) * 28//d, 1):
                            summ += a[k][u][v]
                            maxx = max(maxx, a[k][u][v])
                            minn = min(minn, a[k][u][v])
                if pre == 0: res[k][i][j] = summ // ((28//d)*(28//d))
                if pre == 1: res[k][i][j] = maxx
                if pre == 2: res[k][i][j] = minn
        ans[k] = (res[k].flatten())
    return ans

#def check_label()

X_train, y_train = load_mnist('data/', kind='train')

print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
#fig, ax = plt.subplots(nrows=2, ncols=5, sharex=False, sharey=False,)
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()

#Toan code phan check ket qua tra ve
X_test, y_test = load_mnist('data/', kind='t10k') #load file test
#input la label 
def checkAccurate(checkLabel, testLabel):
    count=0
    for i in range(10000):
        if (checkLabel[i]==testLable[i]):
            count+=1
    return count/10000


"""
# TUAN 2
pp = process(X_train)
his = calchis(pp)
print(his.shape[0])
print(his.shape[1])
#for i in range(len(his)):
#    print(his[i])

# TUAN 3
d, pre = 4, 0 #pre la sum max min 
#ds = downsampling(X_train, d, pre)
ds0 = downsampling(X_train, 4, 0)
ds1 = downsampling(X_train, 4, 1)
ds2 = downsampling(X_train, 4, 2)
print(ds0[0])
print(ds1[0])
print(ds2[2])

# TUAN 4

# TU CODE k
"""

"""
#hien so ra
for i in range(10):
    img = X_train[y_train == i][0]
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
"""