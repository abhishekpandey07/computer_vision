import numpy as np
import sys
import os
import  cv2
import cPickle as pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_data(file):
    f =  open(file,'rb')
    if f is not None:
        X_train  = pickle.load(f)
        Y_train =  pickle.load(f)
        X_test =  pickle.load(f)
        Y_test =  pickle.load(f)
        mean = pickle.load(f)
        return X_train, Y_train, X_test, Y_test, mean
    else:
        print 'File  not Found.'
        raise SystemExit

def main(argv):
    labels =  ['ball','false']
    filename =  argv[1]
    if os.path.isfile(filename):
        X_train,Y_train,X_test,Y_test,mean = load_data(filename)
    else:
        print 'File doesnt exist'
    
    print X_train[50:70]   
    print np.mean(X_train,axis = 0)==mean
    pca = PCA(n_components=2)

    X = pca.fit_transform(X_train)
    for i,(x,y) in enumerate(X):
        for l in labels:
            if Y_train[i]==l:
                if l == 'ball':
                    plt.scatter(x,y,color='r')
                else:
                    plt.scatter(x,y,color='b')
                
    plt.show()
            
if __name__ =='__main__':
    main(sys.argv)
    

    
