import cv2
img = cv2.imread('../images/person_dark.jpg',cv2.IMREAD_REDUCED_GRAYSCALE_2)
s_imageX = cv2.Scharr(img, cv2.CV_8U, 1,0)
s_imageY = cv2.Scharr(img, cv2.CV_8U, 0,1)
cv2.imshow('Original image', img)
cv2.imshow('Scharr X direction image', s_imageX)
cv2.imshow('Scharr Y direction image', s_imageY)
cv2.waitKey(0)