from paz_second.pipelines import DetectMiniXceptionFER
import cv2
import numpy as np
from paz_second.backend.image import BGR2RGB
from PIL import Image
from scipy.spatial import distance
from imutils import face_utils
import dlib

RGB2BGR = cv2.COLOR_RGB2BGR
face_detector = dlib.get_frontal_face_detector()
face_parts_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def overlayImage(src, overlay, location, size):
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    pil_src = Image.fromarray(src)
    pil_src = pil_src.convert('RGBA')

    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGRA2RGBA)
    pil_overlay = Image.fromarray(overlay)
    pil_overlay = pil_overlay.convert('RGBA')
    #顏の大きさに合わせてリサイズ
    pil_overlay = pil_overlay.resize(size)

    # 画像を合成
    pil_tmp = Image.new('RGBA', pil_src.size, (255, 255, 255, 0))
    pil_tmp.paste(pil_overlay, location, pil_overlay)
    result_image = Image.alpha_composite(pil_src, pil_tmp)

    # OpenCV形式に変換
    return cv2.cvtColor(np.asarray(result_image), cv2.COLOR_RGBA2BGRA)

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

if __name__ == "__main__":
    x = 0
    y = 0
    w = 0
    h = 0
    imag1=".\images\main.png"
    imag2=".\images\happy.png"
    imag3=".\images\close.png"
    imag4=".\images\happy_close.png"
    cv_img1 = cv2.imread(imag1,cv2.IMREAD_UNCHANGED)
    cv_img2 = cv2.imread(imag2,cv2.IMREAD_UNCHANGED)
    close_1 = cv2.imread(imag3,cv2.IMREAD_UNCHANGED)
    close_2 = cv2.imread(imag4,cv2.IMREAD_UNCHANGED)


    pipeline = DetectMiniXceptionFER([0.1, 0.1])

    camera = cv2.VideoCapture(0)
    emotion = "neutral"
    while True:
        ret,image = camera.read()
        image = cv2.cvtColor(image,BGR2RGB)
        output = pipeline(image)
        image = output["image"]
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        image,eye = face_landmark_find(image)
        if len(output["boxes2D"]) ==1:
            image = cv2.resize(image,(640,480),interpolation=cv2.INTER_LINEAR)
            x_min, y_min, x_max, y_max = output["boxes2D"][0].coordinates
            w,h = (x_max-x_min,y_max-y_min)
            x,y =x_min,y_min
            emotion = output["boxes2D"][0].class_name
        if w != 0:
            if eye < 0.2:
                image =overlayImage(image,cv_img1,(x,y),(w,h))
                print("目が閉じている")
            elif eye >= 0.2:
                image = overlayImage(image,cv_img2,(x,y),(w,h))

        cv2.imshow("frame", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()
