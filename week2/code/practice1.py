import cv2

if __name__ == '__main__':
    img = cv2.imread('../images/person_light.jpg', cv2.IMREAD_GRAYSCALE)
    _, dst = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow('dst', dst)

    cv2.waitKey(0)