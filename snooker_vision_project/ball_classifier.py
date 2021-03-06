import os
import sys
import numpy as np
import cv2
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import recall_score,f1_score,accuracy_score
import cPickle as pickle
import matplotlib.pyplot as plt

class BallClassifier:
    def __init__(self,model_type = 'lol', model_name = '0',
                 save_dir = 'predictors/models/'):
        self.model_name = model_name
        self.model_type = model_type
        self.save_dir = save_dir+self.model_type+'/'
        if self.model_type == 'SVC':
            self.model = svm.SVC()
        elif self.model_type == 'SGD':
            self.model = SGDClassifier(random_state=7)
        elif self.model_type == 'RFC':
            self.model = RandomForestClassifier()
        elif self.model_type == 'MLP':
            self.model = MLPClassifier()
        #  Add more model later
        else:
            print 'Invalid Model Type'
            raise SystemExit

    def fit(self,X_train,Y_train):
        self.model.fit(X_train,Y_train)
        return self

    def predict(self,y):
        return self.model.predict(y)                
    def transform(self,y):
        return self.predict(y)
    
    def score(self,X,y):
        return self.model.score(X,y)

    def save_model(self):
        if not(os.path.isdir(self.save_dir)):
            os.makedirs(self.save_dir)
            print 'Directory :', self.save_dir, ' created successfully.'
    
        save_file = self.save_dir+self.model_name+'.pkl'
        f = open(save_file,'wb')
        pickle.dump(self.model,f)
        f.close()
        print 'Model saved successfully to ' + save_file

    def load_model(self):
        if not(os.path.isdir(self.save_dir)):
            print 'Load failed. Directory doesnt exist'

        else:
            load_file = self.save_dir+self.model_name+'.pkl'
            f = open(load_file,'rb')
            self.model = pickle.load(f)
            if self.model is not None:
                'Loading successful.'
            else:
                print 'An error occured. Load_failed'

def load_data(file,process):
    f = open(file,'rb')
    mean = None; label_encoder = None;
    scaler = None;
    if f is not None:
        X_train = pickle.load(f)
        Y_train =  pickle.load(f)
        X_test =  pickle.load(f)
        Y_test =  pickle.load(f)
    return  X_train,Y_train,X_test,Y_test
                
def cross_validate(classifier,X,Y,n_splits):
    skfolds = StratifiedKFold(n_splits=n_splits,random_state=7)
    best_accuracy = 0
    clones = []
    accuracies = []
    for train_index,test_index in skfolds.split(X,Y):      
        clone_clf = clone(classifier.model)
        X_train = X[train_index]
        Y_train = Y[train_index]
        X_test = X[test_index]
        Y_test = Y[test_index]
        clone_clf.fit(X_train,Y_train)
        y_pred = clone_clf.predict(X_test)
        accuracy = accuracy_score(Y_test,y_pred,average=None)
        accuracies.append(accuracy)
        clones.append(clone_clf)

    #clones=np.array(clones)
    accuracies = np.array(accuracies)
    print 'accuracies:',precisions
    best_accuracy = 0
    index = -1
    for i,b in enumerate(accuracies):
        if b > best_accuracy:
            index = i
            best_accuracy = b
    best_clone = clones[index]
    print 'Saving best model with accuracy:', best_accuracy
    classifier.model = best_clone
    return accuracies, classifier

def main(argv):
    load_data_file =  argv[1]
    mode = argv[2]
    process = argv[3]
    encode = argv[5]
    scale = argv[4]
    if len(argv) > 2:
        save_model_dir = argv[6]
    model_types  = ['SGD','SVC','RFC','MLP']

    X_train, Y_train,X_test,Y_test = load_data(load_data_file,process)
    for model in model_types:
        print '\nTraining and Evaluating a', model,'classifier'
        bclf = BallClassifier(model_type = model, model_name = process+'_'+mode)
        bclf.fit(X_train,Y_train)
        #Y_pred =  bclf.predict(X_train)
        #conf = confusion_matrix(Y_train,Y_pred)
        #print "Printing Confusion Matrix for",model," classifier."
        #print conf
        #plt.matshow(conf)
        print 'Cross Validating ', model, 'classifier'
        scores, bclf = cross_validate(bclf,X_train,Y_train,3)
        #plt.show()

        print'\nEvaluating  Test set with :'
        y_pred = bclf.predict(X_test)
        conf = confusion_matrix(Y_test,y_pred)
        print conf
        #plt.matshow(conf)
        precision = precision_score(Y_test,y_pred,average='macro')
        recall = recall_score(Y_test,y_pred,average = 'macro')
       # f1 = f1_scrore(Y_test,y_pred)
        accuracy = accuracy_score(Y_test,y_pred)
        print 'Accuracy:', accuracy
        print 'Precision:', precision
        print 'Recall:', recall

        print 'Saving the model to', save_model_dir+model+'/'
        bclf.save_model()
        #print 'F1_score:', f1
        #plt.show()
if  __name__ == '__main__':
    main(sys.argv)
