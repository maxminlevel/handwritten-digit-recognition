import numpy as np
import pickle
def hiscal(X_train):
    #duoi mang
    nX_train = np.zeros((60000,784))
    count=0
    for i in X_train:
        #print(i.flatten().shape[0])
        nX_train[count]=(i.flatten())
        count+=1
    #tinh histogram
    index=0
    his = np.zeros((60000,256))
    for i in nX_train:
        for j in i:
            his[index][int(j)]+=1
        index+=1
    #with open('histogram.pickle', 'wb') as f:
    #    pickle.dump(his, f)
    return his