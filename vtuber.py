from operator import truediv
import cv2 as cv
import dlib
from imutils import face_utils
from scipy.spatial import distance
import random
from msvcrt import getch
import numpy as np
from PIL import Image

WIDTH = 1920
HEIGHT = 1080

face_detector = dlib.get_frontal_face_detector()
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)
face_parts_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')


img1="main.png"
img2="close.png"
img3="happy.png"
img4="ang.png"

cv_img1 = cv.imread(img1,cv.IMREAD_UNCHANGED)
cv_img2 = cv.imread(img2,cv.IMREAD_UNCHANGED)
cv_img3 = cv.imread(img3,cv.IMREAD_UNCHANGED)
cv_img4 = cv.imread(img4,cv.IMREAD_UNCHANGED)

EYE_AR_THRESH = 0.2



def overlayImage(src, overlay, location, size):
    overlay_height, overlay_width = overlay.shape[:2]

    src = cv.cvtColor(src, cv.COLOR_BGR2RGB)
    pil_src = Image.fromarray(src)
    pil_src = pil_src.convert('RGBA')

    overlay = cv.cvtColor(overlay, cv.COLOR_BGRA2RGBA)
    pil_overlay = Image.fromarray(overlay)
    pil_overlay = pil_overlay.convert('RGBA')
    #顏の大きさに合わせてリサイズ
    pil_overlay = pil_overlay.resize(size)

    # 画像を合成
    pil_tmp = Image.new('RGBA', pil_src.size, (255, 255, 255, 0))
    pil_tmp.paste(pil_overlay, location, pil_overlay)
    result_image = Image.alpha_composite(pil_src, pil_tmp)

    # OpenCV形式に変換
    return cv.cvtColor(np.asarray(result_image), cv.COLOR_RGBA2BGRA)


def fan(mouth):
    A = distance.euclidean(mouth[0], mouth[9])
    B = distance.euclidean(mouth[6], mouth[9])
    C = distance.euclidean(mouth[3], mouth[9])
    mouth_fan = (A+B)/(C*2)#0.9以上
    return mouth_fan

def calc_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    eye_ear = (A + B) / (2.0 * C)
    return eye_ear

def face_landmark_find(img):
    eye = 10

    img_gry = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_detector(img_gry, 1)

    for face in faces:
        landmark = face_parts_detector(img_gry, face)
        landmark = face_utils.shape_to_np(landmark)

        left_eye_ear = calc_ear(landmark[42:48])
        right_eye_ear = calc_ear(landmark[36:42])
        eye = (left_eye_ear + right_eye_ear) / 2.0

        mouth = fan(landmark[49:60])
    return img,eye,mouth


x = 0
y = 0
w = 0
h = 0

while True:
    ret,rgb = cap.read()
    rgb,eye,mouth = face_landmark_find(rgb)


    gray = cv.cvtColor(rgb,cv.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, 1.3, 5)

    if face != ():
        (x, y, w, h) = face[0]
        x=x-1
        w=w+2
        h=h+2

    if w != 0:
        if mouth >= 0.9:
            rgb = overlayImage(rgb, cv_img3, (x, y), (w, h))
        elif eye < EYE_AR_THRESH:
            rgb = overlayImage(rgb, cv_img2, (x, y), (w, h))
        elif eye >= EYE_AR_THRESH:
            rgb = overlayImage(rgb, cv_img1, (x, y), (w, h))

    cv.imshow("frame",rgb)

    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()