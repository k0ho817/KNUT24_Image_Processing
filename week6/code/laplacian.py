import cv2
img = cv2.imread('../images/person_dark.jpg',cv2.IMREAD_REDUCED_GRAYSCALE_2)
l_image = cv2.Laplacian(img, cv2.CV_8U, ksize=3)
cv2.imshow('Original image', img)
cv2.imshow('Laplacian image', l_image)
cv2.waitKey(0)