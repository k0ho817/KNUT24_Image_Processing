import cv2

if __name__ == '__main__':
    img = cv2.imread('../images/Lenna.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow("Image", img)
    cv2.imwrite('Lenna_1.bmp', img)
    cv2.imwrite('Lenna_2.jpg', img)
    cv2.waitKey(0)