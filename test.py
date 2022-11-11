import os
import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance

face_detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(0)
face_parts_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

def calc_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    eye_ear = (A + B) / (2.0 * C)
    return eye_ear

def face_landmark_find(img):
    eye = 10

    img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(img_gry, 1)

    for face in faces:
        landmark = face_parts_detector(img_gry, face)
        landmark = face_utils.shape_to_np(landmark)

        left_eye_ear = calc_ear(landmark[42:48])
        right_eye_ear = calc_ear(landmark[36:42])
        eye = (left_eye_ear + right_eye_ear) / 2.0

    return img,eye

def setting():
    #f = open('user.txt', 'w')
    count = 0
    eye_data =[]
    while True:
        ret,rgb = cap.read()
        rgb,eye = face_landmark_find(rgb)
        eye_data.append = (eye)
        if ret == True:
            count +=1
        cv2.imshow("image",rgb)
        if(count > 100):
            print(eye_data)
            cap.release()
            cv2.destroyAllWindows()
            break

    #f.close()

def count_file():
    f=open('user.txt')
    count = 0
    for line in f:
        count += 1
    print(count)
    f.close()

r = os.path.exists('user.txt')
print(r) # True (存在する)
if r == False:
    print("初期設定を行います")
    print("目を閉じてください")
    setting()
elif r == True:
    f = open('user.txt')
    lins = f.readlines()
    print(lins[0])
    count_file()
    f.close()
