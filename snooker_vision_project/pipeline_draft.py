import cv2
import numpy as np
import common
from glob import glob
import os
import cPickle as pickle
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.preprocessing import LabelBinarizer, StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline, make_pipeline
import sys

def load_data(label,X,Y):
    count = 0
    base_directory = 'dataset/balls/balls/'
    ext = '.tiff'
    pattern = base_directory + label + '/*' + ext
    print 'Extracting data for label :',label
    files = glob(pattern)
    for f in files:
        print 'Reading file:', f
        img = cv2.imread(f)
        if img is not None:
            if img.shape[0] == 20 and img.shape[1]==25:
                X.append(img)
                Y.append(label)
                count = count +1
    print 'Samples read:',count, 'for', label
    
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
    def  __init__(self,process='gray'):
        self.process = process
        print 'ColorSpaceTransformer: transformation = ',self.process

    def fit(self,X,y=None,**fit_params):
        return self

    def transform(self,X):
        #print 'Transforming colorspace'
        ret = []
        for x in X:
            if self.process == 'hsv':
                x = cv2.cvtColor(x,cv2.COLOR_BGR2HSV)
                x,_,_ = cv2.split(x)
            elif self.process == 'gray':
                x = cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)
            else:
                print 'Invalid Transform. Please enter "gray" or "hsv".'
                raise SystemExit
            ret.append(x.flatten())
        return np.array(ret)

class HogDescriptor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.hog = self.createHogDescriptor()
        if self.hog is not None:
            print 'HogDescriptor Ready.'
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
    
    def fit(self,X,y=None,**fit_params):
        return self
    def transform(self,X):
        #print 'Creating Hog Descriptor'
        ret = []
        for x in X:
            x = cv2.resize(x,(20,20))
            x = self.hog.compute(x)
            ret.append(x.flatten())
        return np.array(ret)

class LabelTransform(BaseEstimator,TransformerMixin):
    def __init__(self, mode = 'ovo'):
        self.mode = mode
        print 'Label Transformer: mode = ', self.mode

    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        if self.mode == 'ovo':
            return X
        elif self.mode == 'ova':
            for i,x in enumerate(X):
                if x != 'false':
                    X[i] = 'ball'
            return X
        else:
            print 'Invalid mode selection. Pass "ovo" or "ova".'
            raise SystemExit

    def inverse_transform(self,X):
        return X
    
class labelencoder(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.encoder = LabelEncoder()
    def fit(self,X,y=None,**fit_params):
        self.encoder.fit(X)
        return self
    def transform(self,X):
        return self.encoder.transform(X)
    def inverse_transform(self,X):
        return self.encoder.inverse_transform(X)
    
def createHogPipeline():
    return Pipeline([
            ('color_transform',ColorSpaceTransformer('gray')),
            ('hog_descriptor',HogDescriptor())
            ])

def createPipelines(mode,process):
    label_pipeline = Pipeline([
        ('l_trans',LabelTransform(mode)),
        ('encoder',labelencoder())
    ])
    
    if process != 'hog':
        samples_pipeline = Pipeline([
            ('color_transform',ColorSpaceTransformer(process)),
            ('scaler',StandardScaler())
            ])
    else:
        print 'Process = "hog". Skipping StandardScaler.'
        samples_pipeline = createHogPipeline()

    return samples_pipeline,label_pipeline

def main(argv):
    
    data_save_directory = 'dataset/balls/processedDataPipeline/'
    pipeline_save_directory = 'dataset/balls/pipelines/'
    save_file = argv[1]
    mode = argv[2]
    process = argv[3]
    print 'Save_file:', save_file
    print 'Mode:', str(mode)
    print 'Process:', str(process)
    
    labels = ['red','yellow','green','blue','brown',
              'pink','black','cue_ball','false']

    X = []
    Y = []
    for l in labels:
        load_data(l,X,Y)
    
    X = np.array(X)
    Y = np.array(Y)

    sample_pipeline,label_pipeline = createPipelines(mode,process)
    
    Y = label_pipeline.fit_transform(Y)
    print 'Labels Transformed'
    X = sample_pipeline.fit_transform(X)
    print 'Data transformed'
    X_train,Y_train,X_test,Y_test = split_stratified(X,Y)
    print 'Training Data :  ',len(X_train)
    print 'Testing Data :  ', len(X_test)
    
    if not(os.path.isdir(data_save_directory)):
        os.makedirs(data_save_directory)
    if not(os.path.isdir(pipeline_save_directory)):
        os.makedirs(pipeline_save_directory)
    
    file_name = data_save_directory + save_file
    print 'Saving data to :', file_name
    f = open(file_name, 'wb')
    pickle.dump(X_train,f)
    pickle.dump(Y_train,f)
    pickle.dump(X_test,f)
    pickle.dump(Y_test,f)
    f.close()

    file_name = pipeline_save_directory+'pipeline-'+save_file
    print 'Saving pipelines to:', file_name
    f = open(file_name,'wb')
    if process != 'hog':
        pickle.dump(sample_pipeline,f)
    else:
        print "can't save HOGDescriptor object."
        print 'Use module to create instances and pipelines'
    pickle.dump(label_pipeline,f)
    f.close()
    
if  __name__ == '__main__':
    main(sys.argv)
    
