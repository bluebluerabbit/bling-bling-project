# import
import cv2
import timeit
import numpy as np
import function as fn
from PIL import Image, ImageFilter, ImageOps
import matplotlib.pylab as pylab

# 함수
def kill_bg(frame, bg):
    fgmask = bg.apply(frame)
    _,fgmask = cv2.threshold(fgmask, 175, 255, cv2.THRESH_BINARY)
    results = cv2.findContours(
    fgmask , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in results[0]:
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)


# 웹캠 입력
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 이미지 입력
image = cv2.imread('image.jpg')
image = cv2.resize(image, (640, 480))

# 이미지 적용
scene = cv2.imread('640x480-image.jpg')
scene = cv2.resize(scene, (640, 480))


# 프레임 복사본 생성 후, salt pepper noise를 제거한다.
cloneImage = image.copy()  # 현재 프레임의 복사본 생성
grayImage = cv2.cvtColor(cloneImage, cv2.COLOR_BGR2GRAY)  # 복사본을 회색조 영상으로 변환
grayImage = cv2.medianBlur(grayImage, 9)  # 블러링으로 노이즈 제거 -> 추후 수정 필요

# 커널 생성
kernel = np.ones((2, 2))
thresholdImage = cv2.threshold(grayImage, 127, 255, cv2.THRESH_OTSU)[1]  # 공부 필요
thresholdImage = cv2.adaptiveThreshold(thresholdImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                       cv2.THRESH_BINARY, 9, 5)
thresholdImage = cv2.dilate(thresholdImage, kernel, iterations=6)  # 팽창
thresholdImage = cv2.erode(thresholdImage, kernel, iterations=20)  # 침식

cloneCopy = cloneImage.copy()  # 복사본의 복사본 생성
cloneCopy[thresholdImage != 0] = scene[thresholdImage != 0]  # 임계치 적용

# 임계치 적용 이미지 출력
cv2.imshow("Thresh Result", thresholdImage)

# edge 검출 함수
edges = cv2.Canny(grayImage, 30, 100)

# 커널 생성 후 edge 팽창, 침식
kernel = np.ones((4, 4))
edges = cv2.dilate(edges, kernel, iterations=5)
edges = cv2.erode(edges, kernel, iterations=2)

# edges 임계치 적용 이미지 출력
cv2.imshow("Canny result", edges)

(cnts, _) = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

dm = np.zeros_like(edges)
if len(cnts) > 0:
    mcnt = max(cnts[:], key=cv2.contourArea)
    dm = cv2.fillConvexPoly(dm, mcnt, (255))
    cv2.imshow("DM", dm)
c = image.copy()
c[dm != 255] = scene[dm != 255]
cv2.imshow("Canny Result", c)

cv2.waitKey(0)


'''
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 웹캠 창 크기
wCam, hCam = 640, 480

while cap.isOpened():  # 웹캠이 켜진 동안 실행
    success, image = cap.read()

    if not success:
        print("Ignoring empty camera frame.")
        # 비디오 : break / 웹캠 : continue
        continue

    # 프레임 복사본 생성 후, salt pepper noise를 제거한다.
    cloneImage = image.copy() # 현재 프레임의 복사본 생성
    grayImage = cv2.cvtColor(cloneImage, cv2.COLOR_BGR2GRAY) # 복사본을 회색조 영상으로 변환
    grayImage = cv2.medianBlur(grayImage, 9) # 블러링으로 노이즈 제거 -> 추후 수정 필요

    # 커널 생성
    kernel = np.ones((2, 2))
    thresholdImage = cv2.threshold(grayImage, 127, 255, cv2.THRESH_OTSU)[1] # 공부 필요
    thresholdImage = cv2.adaptiveThreshold(thresholdImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
                                      cv2.THRESH_BINARY, 9, 5)
    thresholdImage = cv2.dilate(thresholdImage, kernel, iterations=6) # 팽창
    thresholdImage = cv2.erode(thresholdImage, kernel, iterations=20) # 침식

    cloneCopy = cloneImage.copy() # 복사본의 복사본 생성
    cloneCopy[thresholdImage!=0] = scene[thresholdImage!=0] # 임계치 적용

    # 임계치 적용 이미지 출력
    cv2.imshow("Thresh Result", thresholdImage)

    # edge 검출 함수
    edges = cv2.Canny(grayImage, 30, 100)

    # 커널 생성 후 edge 팽창, 침식
    kernel = np.ones((4, 4))
    edges = cv2.dilate(edges, kernel, iterations=5)
    edges = cv2.erode(edges, kernel, iterations=5)

    # edges 임계치 적용 이미지 출력
    cv2.imshow("Canny", edges)

    (cnts, _) = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    dm = np.zeros_like(edges)
    if len(cnts)>0:
        mcnt = max(cnts[:], key=cv2.contourArea)
        dm=cv2.fillConvexPoly(dm, mcnt, (255))
        cv2.imshow("DM", dm)
    c = image.copy()
    c[dm!=255]=scene[dm!=255]
    cv2.imshow("Canny Result", c)


    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 창 이름, 위치 설정
    winname = "Let's Bling-Bling!" # 창 이름 지정
    #cv2.namedWindow(winname) # 창 이름 적용
    #cv2.moveWindow(winname, 640, 300) # 웹캠 위치

    start_t = timeit.default_timer()

    # 영상처리
    fps = cap.get(cv2.CAP_PROP_FPS)  # 프레임 수
    if fps != 0:
        delay = int(1000 / fps)

    bg = cv2.createBackgroundSubtractorMOG2()
    kill_bg(image, bg)



    # mask = bg.apply(image)
    # cv2.imshow('mask', mask)
    
    #cv2.imshow(winname, cv2.flip(image, 1))
    if cv2.waitKey(1) & 0xFF == 27:
        break
    cv2.waitKey(1)

cap.release()
'''


# 영상처리

if __name__ == '__main__':
    print('Bling-Bling!')