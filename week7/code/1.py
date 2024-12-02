import cv2

capture = cv2.VideoCapture(0)
# capture = cv2.VideoCapture('images/moving_light.mp4')

width = capture.get(cv2.CAP_PROP_FRAME_WIDTH) # 입력 영상 width 불러오기
height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) # 입력 영상 height 불러오기

print('Frame width; {}, height {}'.format(width, height))

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1024) # 출력 영상 width 설정
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 768) # 출력 영상 height 설정

while cv2.waitKey(32) < 0: # 32ms 마다 반복 약 30fps
    ret, frame = capture.read() # 프레임 획득 
    if not ret: # ret == False: ret에 값이 False라면
        break
    cv2.imshow("Frame", frame)

capture.release() # 장치 해제