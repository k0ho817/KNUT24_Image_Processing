import numpy as np
import cv2
img = cv2.imread('./images/cameraman.jpg',cv2.IMREAD_REDUCED_GRAYSCALE_2)

roberts_x = np.array([[0, 0,-1], [0, 1, 0], [0, 0, 0]]) #/
roberts_y = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]]) #\

prewitt_x = np.array([[1, 0,-1], [1, 0,-1], [1, 0,-1]]) #| 수직엣지 검출 시험출제
prewitt_y = np.array([[-1,-1,-1], [0, 0, 0], [1, 1, 1]]) #- 수평엣지 검출 시험출제

r_imageX = cv2.convertScaleAbs(cv2.filter2D(img,-1, roberts_x))
r_imageY = cv2.convertScaleAbs(cv2.filter2D(img,-1, roberts_y))

p_imageX = cv2.convertScaleAbs(cv2.filter2D(img,-1, prewitt_x))
p_imageY = cv2.convertScaleAbs(cv2.filter2D(img,-1, prewitt_y))

cv2.imshow('Original image', img)
cv2.imshow('Roberts X direction image', r_imageX)
cv2.imshow('Roberts Y direction image', r_imageY)
cv2.imshow('Prewitt X direction image', p_imageX)
cv2.imshow('Prewitt Y direction image', p_imageY)
cv2.waitKey(0)