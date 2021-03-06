import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import common
import ball_tracker
import sys
from sklearn.neural_network import MLPClassifier
import cPickle as pickle
from ball_classifier import BallClassifier

def grabBalls(img,circles):
    balls = []
    for [(x,y),r] in circles:
        x = x - 15
        y = y - 10
        w = 25
        h = 20
        balls.append(img[y:y+h,x:x+w])
    return balls
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

                
def changePerspective(points, img):
        r , c  = 480, 960
        setpoints = np.float32([[c,r],[c,0],[0,0],[0,r]])
        M = cv2.getPerspectiveTransform(np.float32(self.points),setpoints)
        X = cv2.warpPerspective(img,M,(c,r))
        return X

def getDilatedMask(img, background, method = 'lol'):
    if method == 'subtractThenGray':
            diff = cv2.subtract(img,background)
            diff = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    else:
            background = cv2.cvtColor(background,cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            diff = cv2.subtract(gray,background)
            
    _, mask = cv2.threshold(diff,60,255,cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    return (img,cv2.dilate(mask, kernel, iterations=2))



def process_frame(frame,background,points):
     dst = changePerspective(points,frame)
     #background = changePerspective(points,background)
     mask = getDilatedMask(dst,background)
     cv2.imshow('mask',mask)
     circles = ball_tracker.detectBallContour(dst,mask)
     return dst, circles

def main(argv):
    video = argv[1]
    back_address = argv[2]
    extract = False
    background = cv2.imread(back_address)
    if len(argv) == 4:
        h = argv[3]
    else:
        h = None
    cv2.imshow('back',background)
    cv2.waitKey(0)
    _,_, points = findAndDrawContours(background.copy(),h)
    background = changePerspective(points,background)
    cap = cv2.VideoCapture(video)
    while True:
        ret, frame = cap.read()
       # frame.resize((480,960))
        cv2.imshow('frame',frame)
        dst, circles =process_frame(frame,background,points)
        circle = np.zeros(dst.shape,np.uint8)
        for [(x,y),r] in circles:
                cv2.circle(circle,(x,y),r,(255,255,255),1)
                
        cv2.imshow('changedperspective',dst)
        cv2.imshow('only_circles',circle)
        k = cv2.waitKey(5)
        if k == 27:    
            break
        if k == ord('e'):
            extract = not(extract)
    
if __name__ == '__main__':
    main(sys.argv)
    
    
