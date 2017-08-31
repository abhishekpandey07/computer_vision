import cv2
import numpy as np
import common
from glob import glob
import os
import cPickle as pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer, StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
import sys

def encodeLabels(labels):
    #encoder = LabelBinarizer(sparse_output = False)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    return labels, encoder

def preprocessData(samples,scale = False):
    mean = np.mean(samples,axis = 0)
    samples = np.subtract(samples,mean)
    if scale:
        scaler =  StandardScaler()
        scaler.fit_transform(samples)
        return samples,mean,scaler
    else:
        return samples,mean,None

def load_data(label,X,Y,mode='ovo',process = 'hsv'):
    count =  0
    base_directory = 'dataset/balls/balls/'
    ext = '.tiff'
    pattern = base_directory + label + '/*' + ext
    files = glob(pattern)
    if process == 'hog':
        hog = createHogDescriptor()
    for f in files:
        img = cv2.imread(f)
        if process =='hsv':
            print 'converting ' + f + ' in HSV'
            hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            x,_,_ = cv2.split(hsv)
            x = x.flatten()
        elif process=='gray':
            print 'converting ' + f + ' in Gray' 
            x = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).flatten()

        elif process=='hog':
            x = cv2.cvtColor(cv2.resize(img,(20,20)),cv2.COLOR_BGR2GRAY)
            x = hog.compute(x)
            x = x.flatten()
        else:
            print 'Unknown data process. Please pass "hsv" or "gray"'
            raise SystemExit
        if len(x) == 500 or process == 'hog':
            X.append(x.flatten())
            if mode == 'ova':
                if label !=  'false':
                    Y.append('ball')
                else:
                    Y.append(label)
            elif mode == 'ovo':
                Y.append(label)



def split_data(samples,labels,test_ratio):
    shuffled_indices = np.random.permutation(len(samples))
    test_size  =  int(len(samples)*test_ratio)
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]
    X_train = samples[train_indices]
    Y_train = labels[train_indices]
    X_test = samples[test_indices]
    Y_test = labels[test_indices]
    return X_train,Y_train,X_test,Y_test

def split_stratified(samples,labels):
    skfolds = StratifiedKFold(n_splits = 5, random_state=42)
    for train_index, test_index in skfolds.split(samples,labels):
        X_train = samples[train_index]
        Y_train = labels[train_index]
        X_test = samples[test_index]
        Y_test = labels[test_index]

    return X_train,Y_train,X_test,Y_test

def labelParticipation(Y, labels):
    participation = np.zeros((1,len(labels)))
                             
    for y in Y:
        for i,_ in enumerate(labels):
            if y == labels[i]:
                participation[0][i] = participation[0][i] + 1

    for i,_ in enumerate(labels):
        print labels[i], ' : ', participation[0][i]

class ColorSpaceTransformer(BaseEstimator, TransformerMixin):
    def  __init__(self,transform):
        self.transform = transform 
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        for i,X in enumerate(X):
            if self.transform == 'hsv':
                x = cv2.cvtColor(X,cv2.COLOR_BGR2HSV)
                x,_,_ = cv2.split(x)
            elif self.transform == 'gray':
                x = cv2.cvtColor(X,cv2.COLOR_BGR2GRAY)
            else:
                print 'Invalid Transform. Please enter "gray" or "hsv".'
                raise SystemExit
             X[i] = x.flatten()
        return X

class HogDescriptor(BaseEstimator, TransformerMixin):
    def __init__(self,win_size):
        self.hog = self.creatHogDescriptor()

    def createHogDescriptor(self):
        win_size = (20,20)
        block_size = (10,10)
        block_stride = (5,5)
        cell_size = (5,5)
        nbins =  9
        derivAperture = 1
        winSigma = -1
        histogramNormType = 0
        L2hysthreshold = 0.2
        gammacorrection  =  1
        nlevels = 64
        signedGradients = True
    
        return cv2.HOGDescriptor(win_size,block_size,block_stride,cell_size,nbins,derivAperture,winSigma,histogramNormType,L2hysthreshold,gammacorrection,nlevels, signedGradients)

    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        for x,i in enumerate(X):
            x = cv2.resize(x,(20,20))
            x = self.hog.compute(X)
            X[i] = x.flatten()
        return X

class labelTransform(TransformerMixin):
    def __init__(self, mode = 'ovo'):
        self.mode = mode

    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        if self.mode == 'ovo':
            return X
        elif self.mode == 'ova':
            for x,i in enumerate(X):
                if x != 'false':
                    X[i] = 'ball'
            return X
        else:
            print 'Invalid mode selection. Pass "ovo" or "ova".'
            raise SystemExit
    
def main(argv):
    
    save_directory = 'dataset/balls/processedData/'
    save_file = argv[1]
    mode = argv[2]
    process = argv[3]
    scale = argv[4]
    encode = argv[5]
    print 'Save_Directory:',save_directory
    print 'Save_file:', save_file
    print 'Mode:', mode
    print 'Process:', process
    print 'Scale:', scale
    print 'Encode:', encode
    
    labels = ['red','yellow','green','blue','brown',
              'pink','black','cue_ball','false']

    X = []
    Y = []
    for l in labels:
        load_data(l,X,Y,mode,process)
    
    X = np.array(X)
    Y = np.array(Y)

    if encode == 'True':
        print 'Encoding Labels'
        Y, label_encoder = encodeLabels(Y)
        
    X_train,Y_train,X_test,Y_test = split_stratified(X,Y)
    print 'Training Data :  ',len(X_train)
    print 'Testing Data :  ', len(X_test)

    if mode == 'ova':
        labels = ['false','ball']
    
    print 'Participation of Stratified splitting:'
    labelParticipation(Y_test,labels)

    if process != 'hog':
        print 'Processing Samples. Process is not HOG'
        X_train,mean,scaler = preprocessData(X_train,scale=scale)

    if not(os.path.isdir(save_directory)):
        os.makedirs(save_directory)

    file_name = save_directory + save_file
    print 'Saving data to :', file_name
    f = open(file_name, 'wb')
    pickle.dump(X_train,f)
    pickle.dump(Y_train,f)
    pickle.dump(X_test,f)
    pickle.dump(Y_test,f)
    if encode == 'True':
        print 'Saving Encoder'
        pickle.dump(label_encoder,f)
    if process != 'hog':
        if scaler is not None:
            print ' Saving Scaler'
            pickle.dump(scaler,f)
        pickle.dump(mean,f)
    
if  __name__ == '__main__':
    main(sys.argv)
    
