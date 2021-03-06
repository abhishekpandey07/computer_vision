import cv2
import numpy as np
from glob import glob


global isDraw
isDraw = False

def draw():
        global x,y,w,h
	_ , contours, heirarchy = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours]
	area = 0 
	for x in contours:
		if cv2.contourArea(x) > area:
			area = cv2.contourArea(x)
                        draw = x
	cv2.drawContours(frame,[draw],-1,(255,0,0),3,cv2.LINE_AA)
        x,y,w,h = cv2.boundingRect(draw)
        print x,y,w,h
        cv2.rectangle(frame,(x,y-5),(x+w,y+5+h),(0,0,255),3)
	
        
'''captures = 'dataset/captures/*.tiff'
files = glob(captures)

file = files[4]

img = cv2.imread(file)'''

lower = np.array([0,100,100])
upper = np.array([0,255,255])
cap = cv2.VideoCapture(0)

def nothing(x):
	pass

def changeH(h):
	global lower , upper
	lower.itemset(0,h)
	upper.itemset(0,h+20)
	
def changeS(s):
	global lower , upper
	lower.itemset(1,s)

def changeV(v):
	global lower , upper
	lower.itemset(2,v)

cv2.namedWindow('Control',cv2.WINDOW_NORMAL)	
cv2.createTrackbar('H','Control',0,179,changeH)
cv2.createTrackbar('S','Control',0,255,changeS)
cv2.createTrackbar('V','Control',0,255,changeV)

cap =  cv2.VideoCapture('dataset/Alain-pink.mkv')
while True:
	ret, frame = cap.read()

        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	mask  = cv2.inRange(hsv,lower,upper)

	res = cv2.bitwise_and(frame,frame,mask = mask)
        if isDraw:
                draw()
                roi = frame[y-5:y+5+h,x:x+w]
                if roi.shape[0] > 0 & roi.shape[1] > 0:
                        cv2.imshow('roi',roi)
	cv2.imshow('frame',frame)
	k = cv2.waitKey(5) & 0xFF

	if k ==  ord('x'):
		break

        if k == ord('c'):
              isDraw = not(isDraw)
cap.release()
cv2.destroyAllWindows()
