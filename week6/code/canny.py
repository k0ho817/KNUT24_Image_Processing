import cv2
img = cv2.imread('../images/person_dark.jpg',cv2.IMREAD_REDUCED_GRAYSCALE_2)
c_image1 = cv2.Canny(img, 10, 50)
c_image2 = cv2.Canny(img, 150, 300)
cv2.imshow('Original image', img)
cv2.imshow('Canny image 1', c_image1)
cv2.imshow('Canny image 2', c_image2)
cv2.waitKey(0)