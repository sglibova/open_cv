import cv2
import numpy as np

#importing practice image
#img = cv2.imread("C:/Users/svett/Pictures/Photos/april14.jpg")
img2 = cv2.imread("C:/Users/svett/Downloads/cards.jpg")

#practicing vertical and horizontal stacking
#imgHor = np.hstack((img2, img2))
#imgVer = np.vstack((img2, img2))

#a function to address stacking issues
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


imgStack = stackImages(0.5, ([img2],[img2]))


#print(img.shape)

#Practicing with different resizing and visibility options

#imgResize = cv2.resize(img, (640, 480))
#imgCropped = img[0:600, 400:1600]
#imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)
#imgCanny = cv2.Canny(img, 200, 200)
#imgDilation = cv2.dilate(imgCanny, kernel, iterations=5)
#imgEroded = cv2.erode(imgDilation, kernel, iterations=1)

#cv2.imshow("GrayImage", imgGray)
#cv2.imshow("BlurredImage", imgBlur)
#cv2.imshow("CannyImage", imgCanny)
#cv2.imshow("DilatedImage", imgDilation)
#cv2.imshow("ErodedImage", imgEroded)

#cv2.imshow("ResizedImage", imgResize)
#cv2.imshow("CroppedImage", imgCropped)

#Generating boxes, lines, text

#img = np.zeros((512, 512, 3), np.uint8)
#img[200:300, 200:300] = 255,0,0
#cv2.line(img, (0,0), (img.shape[1], img.shape[0]), (0, 255, 0), 3)
#cv2.rectangle(img, (0,0), (250, 350), (0, 0, 255), 2)
#cv2.circle(img, (400, 50), 30, (255, 255, 0), 5)
#cv2.putText(img, "OpenCV", (300, 200), cv2.FONT_HERSHEY_COMPLEX, .75, (0, 150, 0), 2)

#width, height = 250, 350
#pts = np.float32([[422, 49],[657, 113], [325, 404], [560,467]])
#pts2 = np.float32([[0,0],[width, 0],[0, height],[width, height]])
#matrix = cv2.getPerspectiveTransform(pts, pts2)
#imgOutput = cv2.warpPerspective(img, matrix, (width, height))

#cv2.imshow("Image", img)
#cv2.imshow("Output", imgOutput)

cv2.imshow("Image Stack", imgStack)
cv2.waitKey(0)