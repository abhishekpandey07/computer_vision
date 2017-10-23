import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import common
import ball_tracker
import sys
from sklearn.preprocessing import StandardScaler, LabelBinarizer, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import time
import cPickle as pickle
from ball_classifier import BallClassifier
from pipeline_draft import ColorSpaceTransformer, HogDescriptor, createHogPipeline, LabelTransform,labelencoder

global rectangles_filtered
global changed_img

def max_area_contour(contours):
        maximum_area = 0
        select = None
        for cnt in contours:
            area =  cv2.contourArea(cnt)
            if area  > maximum_area:
                maximum_area = area
                select = cnt
        return select
def findContour(img,h=None):
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        if h is None:
                h  = np.median(hsv[:,:,0].ravel())
        else:
                h = int(h)
        print 'setting lower h channel to ', h
        lower = np.array([h,0,0])
        higher = np.array([h+20,255,255])
       
        mask = cv2.inRange(hsv,lower,higher)
        _,contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            draw = max_area_contour(contours)
            if draw is not None:
                epsilon = 0.009*cv2.arcLength(draw,True)
                approx = cv2.approxPolyDP(draw,epsilon,True)
        return approx

def findAndDrawContours(img,h = None):

        
        approx = findContour(img,h)
        if approx  is not None:
                points = approx.reshape(1,-1)
                points[0][4] = points[0][4] + 1
                points[0][3] = points[0][3] - 3
                points[0][0] = points[0][0] - 3
                points[0][2] = points[0][2] + 4
                points[0][5] = points[0][5] - 5
                points[0][7] = points[0][7] - 5
                points[0][6] = points[0][6] - 3
                points = points.reshape(-1,2)

                #changePerspective(points,img)
                #cv2.drawContours(img,[points],-1,[255,0,0],1)
                #x,y,w,h = cv2.boundingRect(approx)
                #cv2.rectangle(img,(x,y-5),(x+w,y+h),(100,0,100),2)        
                return True, img, points
            
        return False, None, None

class changePerspective(BaseEstimator,TransformerMixin):
    def __init__(self, points):
        self.points = points

    def fit(self,X,y=None,**fit_params):
        return self
    def transform(self,X):
        r , c  = 480, 960
        setpoints = np.float32([[c,r],[c,0],[0,0],[0,r]])
        M = cv2.getPerspectiveTransform(np.float32(self.points),setpoints)
        X = cv2.warpPerspective(X,M,(c,r))
        global changed_img
        changed_img = X
        return X

class getDilatedMask(BaseEstimator,TransformerMixin):
    def __init__(self,background,method='lol'):
        self.background = background
        self.method = method
        if self.method != 'subtractThenGray':
            self.background = cv2.cvtColor(self.background,cv2.COLOR_BGR2GRAY)
        
    def fit(self,X,y=None,**fit_params):
        return self
    def transform(self,img):
        if self.method == 'subtractThenGray':
            diff = cv2.subtract(img,self.background)
            diff = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            diff = cv2.subtract(gray,self.background)
   
        _, mask = cv2.threshold(diff,60,255,cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        return (img,cv2.dilate(mask, kernel, iterations=2))

class GrabBalls(BaseEstimator,TransformerMixin):
    def grabBalls(self,img,circles):
        balls = []
        global rectangles_filtered
        rectangles_filtered = []
        for [(x,y),r] in circles:
            x = x - 15
            y = y - 10
            w = 25
            h = 20
            ball = img[y:y+h,x:x+w]
            if ball.shape[0] == 20 and ball.shape[1] == 25:
                    balls.append(ball)
                    rectangles_filtered.append([x,y,w,h])
        return balls

    def fit(self,X,y=None,**fit_params):
        return self

    def transform(self,(img,mask)):
        circles = ball_tracker.detectBallContour(img,mask)
        balls = self.grabBalls(img,circles)
        return balls

class useLabelPipeline(BaseEstimator,TransformerMixin):
    def __init__(self,pipeline):
        self.pipeline = pipeline

    def fit(self,X,y=None,**fit_params):
        return Self

    def transform(self,X):
        return self.pipeline.inverse_transform(X)

class predictionUnion(BaseEstimator,TransformerMixin):
        def fit(self,X,y=None,**fit_params):
                return self
        def transform(self,predictions):
                length = len(predictions)
                length = length/2
                ball_pred = predictions[:length]
                colour_pred = predictions[length:]
                final = []
                for b,c in zip(ball_pred,colour_pred):
                        if b != 'false' and c != 'false':
                                final.append(c)
                        else:
                                final.append('false')

                return np.array(final)
                                

def main(argv):
    if len(argv) > 1:
        video = argv[1]
        back_address = argv[2]
        h = argv[3]
        if h  == 'None':
            h = None
        else:
            h = int(h)

    else:
        video = 'dataset/video_data/Ronnie/Ronnie-147.mp4'
        back_address = 'dataset/video_data/Ronnie/background.tiff'
        h = 52
        
    pipeline_address = 'dataset/balls/pipelines/pipeline-'
    #mode = str(argv[4])
    #process = str(argv[5])
    pipeline_ball_address = pipeline_address + 'hog_ova.pkl'
    pipeline_colour_address = pipeline_address + 'hsv_ovo.pkl'
    background  = cv2.imread(back_address)
    _,_,points = findAndDrawContours(background.copy(),h)
    perspective =  changePerspective(points)
    background = perspective.fit_transform(background)
    cap = cv2.VideoCapture(video)

    f = open(pipeline_ball_address,'rb')
    data_ball_pipeline = createHogPipeline()
    label_ball_pipeline = pickle.load(f)
    f.close()

    f = open(pipeline_colour_address,'rb')
    data_colour_pipeline = pickle.load(f)
    label_colour_pipeline = pickle.load(f)
    
    if data_colour_pipeline is not None:
        print 'data_colour_pipeline successfully created.'
    if data_ball_pipeline is not None:
        print 'data_ball_pipeline successfully created.'
    if label_colour_pipeline is not None:
        print 'label_colour_pipeline successfully created.'
    if label_ball_pipeline is not None:
        print 'label_ball_pipeline successfully created.'

    ball_classifier = BallClassifier('MLP','hog_ova','predictors/models/')
    ball_classifier.load_model()

    colour_classifier = BallClassifier('RFC','hsv_ovo','predictors/models/')
    colour_classifier.load_model()

    ball_pipeline = Pipeline([('data_ball_pipe',data_ball_pipeline),
                              ('ball_classifier',ball_classifier),
                              ('label_inverse',useLabelPipeline(label_ball_pipeline))
                              ])
    colour_pipeline = Pipeline([('data_colour_pipe',data_colour_pipeline),
                                ('colour_classifier',colour_classifier),
                                ('label_inverse',useLabelPipeline(label_colour_pipeline))
                                ])
    
    frame_pipe = Pipeline([
            ('perspective',perspective),
            ('dilated_mask',getDilatedMask(background,method='lol')),#'subtractThenGray')),
            ('balls',GrabBalls()),
            ('feature_extraction',FeatureUnion([
                    ('ball_pipeline',ball_pipeline),
                    ('colour_pipeline',colour_pipeline)
                    ])),
            ('prediction_union',predictionUnion())
            ])
        
    if frame_pipe is not None:
            print 'frame_pipeline formed'

    extract = False
    start = time.time()
    while True:
        ret, frame = cap.read()
        if time.time() - start >7 and not(extract):
            cv2.putText(frame, "press 'e' to start extracting",(5,20),\
                    cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        cv2.imshow('input',frame)
                
        if extract == True:
                prediction = frame_pipe.transform(frame)
                for predict,[x,y,w,h] in zip(prediction,rectangles_filtered):
                        if predict == 'red':
                                cv2.rectangle(changed_img,(x,y),(x+w,y+h),[0,0,255],2)
                        elif predict == 'black':
                                cv2.rectangle(changed_img,(x,y),(x+w,y+h),[0,0,0],2)
                        elif predict == 'pink':
                                cv2.rectangle(changed_img,(x,y),(x+w,y+h),[255,0,255],2)
                        elif predict == 'blue':
                                cv2.rectangle(changed_img,(x,y),(x+w,y+h),[255,0,0],2)
                        elif predict == 'brown':
                                cv2.rectangle(changed_img,(x,y),(x+w,y+h),[42,42,165],2)
                        elif predict == 'green':
                                cv2.rectangle(changed_img,(x,y),(x+w,y+h),[100,255,0],2)
                        elif predict == 'yellow':
                                cv2.rectangle(changed_img,(x,y),(x+w,y+h),[0,255,255],2)
                        elif predict == 'cue_ball':
                                cv2.rectangle(changed_img,(x,y),(x+w,y+h),[255,255,255],2)
                text =  'reds:'+ str(len(prediction[np.where(prediction == 'red')]))
                cv2.putText(changed_img, text,(5,20),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                cv2.imshow('predicted result', changed_img)
    
        k = cv2.waitKey(1000/100)
        if k == 27:
                break
        elif k == ord('e'):
                extract =  not extract
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main(sys.argv)
