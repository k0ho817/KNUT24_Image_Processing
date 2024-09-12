import cv2

if __name__ == '__main__':
    img = cv2.imread('../images/Lenna.jpg', cv2.IMREAD_UNCHANGED)
    height, width, channels = img.shape
    print(f'height : {height}, width : {width}, channel : {channels}')

    img = cv2.imread('./Lenna_2.jpg', cv2.IMREAD_UNCHANGED)
    height, width = img.shape
    print(f'height : {height}, width : {width}')