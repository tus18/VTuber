from fer import FER
import matplotlib.pyplot as plt
test_image_one = plt.imread("./ang_imgs\cut_image_drink1.jpg")
emo_detector = FER(mtcnn=True)
captured_emotion = emo_detector.detect_emotions(test_image_one)
print(captured_emotion)
dominant_emotion, emotion_score = emo_detector.top_emotion(test_image_one)
plt.imshow(test_image_one)
print(dominant_emotion, emotion_score)