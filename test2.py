from fer import FER
import matplotlib.pyplot as plt
import cv2 as cv
#処理が遅すぎる
WIDTH = 1920
HEIGHT = 1080

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)

while True:
    ret,rgb = cap.read()
    #test_image_one = plt.imread("./ang_imgs\cut_image_drink1.jpg")
    emo_detector = FER(mtcnn=True)
    captured_emotion = emo_detector.detect_emotions(rgb)
    print(captured_emotion)
    dominant_emotion, emotion_score = emo_detector.top_emotion(rgb)
    cv.imshow("flame",rgb)
    print(dominant_emotion, emotion_score)

    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()