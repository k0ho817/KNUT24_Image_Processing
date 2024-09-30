import cv2

src = cv2.imread('./Lenna_2.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('src', src)

_, dst = cv2.threshold(src, 160, 255, cv2.THRESH_BINARY)
cv2.imshow('dst', dst)

cv2.waitKey(0)

print('this is testhjkkjhlkhlkj