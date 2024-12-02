import cv2
import numpy as np

# 1. 입력 이미지 읽기
# 기본 이미지와 검색 대상 이미지
roi_image = cv2.imread('week12/ex7/Einstein.jpg')  # 관심 영역(ROI)이 포함된 이미지
# roi_image = cv2.imread('car1.jpg')  # 관심 영역(ROI)이 포함된 이미지
target_image = cv2.imread('week12/ex7/Einstein.jpg')  # 검색 대상 이미지
# target_image = cv2.imread('car2.jpg')  # 검색 대상 이미지

if roi_image is None or target_image is None:
    print("이미지를 읽을 수 없습니다. 파일 경로를 확인하세요.")
    exit()

# 2. ROI(관심 영역) 설정
roi = cv2.selectROI("Select ROI", roi_image, showCrosshair=True, fromCenter=False)
x, y, w, h = map(int, roi)
roi_cropped = roi_image[y:y+h, x:x+w]  # 관심 영역 자르기

cv2.destroyWindow("Select ROI")  # ROI 선택 창 닫기

# 3. ROI의 HSV 히스토그램 계산
roi_hsv = cv2.cvtColor(roi_cropped, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([roi_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])  # Hue-Saturation 히스토그램
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)  # 히스토그램 정규화

# 4. 검색 대상 이미지에 대해 히스토그램 역투영 수행
target_hsv = cv2.cvtColor(target_image, cv2.COLOR_BGR2HSV)
back_proj = cv2.calcBackProject([target_hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)

# 5. 역투영 결과를 시각적으로 표현
# 결과를 원본 이미지와 비교 가능하도록
result = cv2.merge((back_proj, back_proj, back_proj))  # 채널 합치기 (시각화용)

# 역투영 강조: 마스크를 통해 주요 영역만 표시
_, mask = cv2.threshold(back_proj, 50, 255, cv2.THRESH_BINARY)
highlighted_result = cv2.bitwise_and(target_image, target_image, mask=mask)

# 6. 결과 시각화
cv2.imshow("Original ROI", roi_cropped)
cv2.imshow("Back Projection", result)
cv2.imshow("Highlighted Result", highlighted_result)

cv2.waitKey(0)
cv2.destroyAllWindows()