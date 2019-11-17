import numpy as np
import pickle
def nflatten(X_train):
    #duoi mang
    nX_train = np.zeros((60000,784))
    count=0
    for i in X_train:
        nX_train[count]=(i.flatten())
        count+=1
    #tinh histogram
    return nX_train
def hiscal(X_train):
    #duoi mang
    nX_train = nflatten(X_train)
    index=0
    his = np.zeros((60000,256))
    for i in nX_train:
        for j in i:
            his[index][int(j)]+=1
        index+=1
        break
    #with open('histogram.pickle', 'wb') as f:
    #    pickle.dump(his, f)
    return his
def down_sample(X_train,x,t):
    indexi=indexj=0
    nX_train = np.zeros((X_train.shape[0],int(X_train.shape[1]/x),int(X_train.shape[2]/x)))
    for h in range(X_train.shape[0]):
        for i in range(0,X_train.shape[1],x):
            for j in range(0,X_train.shape[2],x):
                maxx=X_train[h][i][j]
                minn=X_train[h][i][j]
                summ=0
                for ii in range(i,i+x):
                    for jj in range(j,j+x):
                        summ+=X_train[h][ii][jj]
                        maxx=max(maxx,X_train[h][ii][jj])
                        minn=min(minn,X_train[h][ii][jj])
                if t=='max':
                    nX_train[h][indexi][indexj]=maxx
                if t=='min':
                    nX_train[h][indexi][indexj]=minn
                if t=='avg':
                    nX_train[h][indexi][indexj]=summ//(x*x)
                indexj+=1
            indexj=0
            indexi+=1
        indexi=0          
    return nX_train

#test
#a = np.arange(32).reshape(2,4,4)
#print(a)
#print(down_sample(a,2,'avg'))
                