import cv2
import numpy as np
from glob import glob
global hsv

global buttondown
buttondown = False
global img
global background
global edit
global  x0,y0,wf,wh

def printvalue(event,x,y,flags,param):
    global buttondown, x0,y0,w,h,edit, wf,wh
    if event == cv2.EVENT_LBUTTONDOWN:
        x0,y0 = x,y
        print x0,y0
        buttondown  = True
    if event == cv2.EVENT_MOUSEMOVE:
        if buttondown:
            w = np.abs(x-x0)
            h = np.abs(y-y0)
    if event == cv2.EVENT_LBUTTONUP:
        wf = w
        wh = h
        edit = img[y0:y0+wh,x0:x0+wf]

        
img = cv2.imread('dataset/alain_background-1.tiff')
background = cv2.imread('dataset/alain_background-3.tiff')
cv2.namedWindow('background')
cv2.setMouseCallback('background',printvalue)

canedit = False

while(1):
    cv2.imshow('background',background)
    cv2.imshow('img',img)
    if canedit:
        edit_back =  background.copy()
        edit_back[y0:y0+wh,x0:x0+wf]  = edit
        cv2.imshow('edit_back',edit_back)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    if k == ord('c'):
        cv2.imwrite('dataset/alain_background_final.tiff',edit_back)
        print 'file saved'
    if k == ord('e'):
        canedit = not(canedit)
cv2.destroyAllWindows()
