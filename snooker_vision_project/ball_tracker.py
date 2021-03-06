import cv2
import numpy as np
import matplotlib as plt
import common
import work_frame
from find_roi import Snooker

def detectBalls(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([52,0,0])
        higher = np.array([72,255,255])
        mask = cv2.inRange(hsv,lower,higher)
        mask_inv = cv2.bitwise_not(mask)
        return mask_inv

def detectHough(img,mask):
    circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1.2,10,
                                   param1=50,param2=7,minRadius=0,
                                   maxRadius = 15)
    print len(circles[0])
    for i  in circles[0,:]:
        cv2.circle(img,(i[0],i[1]),i[2],(255,0,0),2)
    return circles

def detectBallContour(img,mask):
    _,contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    circles = []
    for cnt in contours:
        (x,y), r = cv2.minEnclosingCircle(cnt)
        x = int(x)
        y = int(y)
        r = int(r)
        if r < 20:
            if r < 7:
                r = 7
            circles.append([(x,y),r])
            #cv2.circle(img,(int(x),int(y)),int(r),(255,0,0),1)
        
       # x0 = int(x) - 15
       # y0 = int(y) - 10
       # x1 = int(x) + 10
       # y1 = int(y) + 10

        #cv2.rectangle(img,(x0,y0),(x1,y1),(255,0,0),1)
    return circles 

def main():
    files = common.getFiles(pattern = 'dataset/captures/*.tiff')
    file = files[2]
    img = cv2.imread(file)
    background = cv2.imread('dataset/background.tiff')
    success,_ , points = work_frame.findAndDrawContours(background.copy())
    snooker = Snooker()
    if success:    
        dst = work_frame.changePerspective(points,img)
        dst1 = dst.copy()
        back = work_frame.changePerspective(points,background)
        dst_gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
        back_gray = cv2.cvtColor(back,cv2.COLOR_BGR2GRAY)
        diff = cv2.subtract(dst,back)
        diff_gray = cv2.subtract(dst_gray,back_gray)
        diff_gray1 = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
        #cv2.imshow('diff_gray',diff_gray)
        #cv2.imshow('diff',diff)
        _ , mask_inv = cv2.threshold(diff_gray,60,255,cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        dilate = cv2.dilate(mask_inv,kernel,iterations = 2)
        #dilate = mask_inv
        circles_cont = detectBallContour(dst1,dilate)
        circles_hough = detectHough(dst,dilate)
      
        cv2.imshow('dst_hough',dst)
        cv2.imshow('dst_contour',dst1)
        cv2.imshow('dilate',dilate)
        cv2.waitKey(0)
    
if __name__ == '__main__':
    main()
