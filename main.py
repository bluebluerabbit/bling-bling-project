# import
import cv2
import numpy as np

# 웹캠 입력
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 이미지 입력
image = cv2.imread('image.jpg')
image = cv2.resize(image, (640, 480))

# 이미지 적용
scene = cv2.imread('640x480-image.jpg')
scene = cv2.resize(scene, (640, 480))

# 이미지 복사본 생성 후, salt pepper noise를 제거한다.
cloneImage = image.copy()  # 현재 이미지의 복사본 생성
grayImage = cv2.cvtColor(cloneImage, cv2.COLOR_BGR2GRAY)  # 복사본을 회색조 영상으로 변환
grayImage = cv2.medianBlur(grayImage, 9)  # 블러링으로 노이즈 제거

# 노이즈가 제거된 이미지 출력
cv2.imshow("image Result", image)

# 커널 생성
kernel = np.ones((2, 2))
thresholdImage = cv2.threshold(grayImage, 127, 255, cv2.THRESH_OTSU)[1]
thresholdImage = cv2.adaptiveThreshold(thresholdImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
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
cv2.imshow("Canny", edges)

# image 외곽선 정보 저장
(cnts, _) = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

dm = np.zeros_like(edges)
if len(cnts) > 0:
    # 배경 : 0, 전경 : 255 => 배경과 전경이 분리된 mask image를 그림
    mcnt = max(cnts[:], key=cv2.contourArea)
    dm = cv2.fillConvexPoly(dm, mcnt, (255))
    cv2.imshow("DM", dm)
c = image.copy()
# 배경 : 0, 전경 : 255
c[dm != 255] = scene[dm != 255] # 전경이 아닌 경우에만 배경 이미지의 화소를 대입함
cv2.imshow("Result", c)

cv2.waitKey(0)

if __name__ == '__main__':
    print('Bling-Bling!')