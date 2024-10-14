import numpy as np
import cv2
if __name__ == '__main__':
    ori_img = cv2.imread("./images/Lenna.jpg", cv2.IMREAD_COLOR)
    rows, cols = ori_img.shape[:2]  # channel 여부 무시
    # pts1 좌표 표시
    pts1 = np.float32([[80, 280], [400, 220], [250, 480], [60, 420]])
    cv2.circle(ori_img, (80, 280), 9, (255, 0, 0), -1)
    cv2.circle(ori_img, (400, 220), 9, (0, 255, 0), -1)
    cv2.circle(ori_img, (250, 480), 9, (0, 0, 255), -1)
    cv2.circle(ori_img, (60, 420), 9, (0, 255, 255), -1)
    cv2.line(ori_img, (0, 340), (511, 340), (255, 0, 0), 2)
    cv2.line(ori_img, (0, 380), (511, 380), (0, 0, 255), 2)
    pts2 = np.float32([[10, 10], [502, 10], [502, 502], [10, 502]])
    Mat1 = cv2.getPerspectiveTransform(pts1, pts2)
    print('Perspective matrix')
    print(Mat1)
    r_image = cv2.warpPerspective(ori_img, Mat1, (cols, rows))
    cv2.imshow('Original image', ori_img)
    cv2.imshow('Perspective image', r_image)
    cv2.waitKey(0)