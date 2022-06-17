# import
import tkinter as tk
import tkinter.messagebox
from tkinter import StringVar, Label, IntVar
import cv2
import timeit

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
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


# 웹캠 창 크기
wCam, hCam = 640, 480

while cap.isOpened():  # 웹캠이 켜진 동안 실행
    success, image = cap.read()

    if not success:
        print("Ignoring empty camera frame.")
        # 비디오 : break / 웹캠 : continue
        continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 창 이름, 위치 설정
    winname = "Let's Bling-Bling!" # 창 이름 지정
    cv2.namedWindow(winname) # 창 이름 적용
    cv2.moveWindow(winname, 640, 300) # 웹캠 위치

    start_t = timeit.default_timer()

    # 영상처리
    fps = cap.get(cv2.CAP_PROP_FPS)  # 프레임 수
    if fps != 0:
        delay = int(1000 / fps)

    bg = cv2.createBackgroundSubtractorMOG2()
    kill_bg(image, bg)

    terminate_t = timeit.default_timer()


    '''
    mask = bg.apply(image)

    cv2.imshow('mask', mask)
    '''
    cv2.imshow(winname, cv2.flip(image, 1))
    if cv2.waitKey(1) & 0xFF == 27:
        break
    cv2.waitKey(1)

cap.release()


# 영상처리

if __name__ == '__main__':
    print('Bling-Bling!')