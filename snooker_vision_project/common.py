from glob import glob
import matplotlib.pyplot as plt
import cv2

def getFiles(pattern='dataset/captures/*.tiff'):
    files = glob(pattern)
    return files

def explore_histogram(img, colour = True, xlim = 256):
    hist = []
    if colour:
        col = ['b','g','r']
        for i, col in enumerate(col):
            hist.append(cv2.calcHist([img],[i],None,[xlim],[0,xlim]))
     #       plt.plot(hist, colour = col)
     #       plt.xlim([0,xlim])
    else:
        hist.append(cv2.calcHist([img],[0],None,[256],[0,xlim]))
     #   plt.plot(hist)
     #   plt.xlim([0,xlim])
    #plt.show()
    return  hist

def explore_histogram2d(img):
    return [cv2.calcHist([img],[0,1],None,[180,250],[0,180,0,256])]

def createGrid(img,size = (8,8)):
    shape  = img.shape
    r = size[0]
    c = size[1]
    row = 0
    col = 0
    while row < shape[0]:
        cv2.line(img,(0,row),(shape[1],row),(0,0,255),2)
        row = row+r
    
    while col < shape[1]:
        cv2.line(img,(col,0),(col,shape[0]),(0,0,255),2)
        col = col+c

    return img
        
