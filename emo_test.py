import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
#確認画面が表示される、感情の結論だけを抽出する必要がある
img = cv2.imread(".\imgs\cut_image_drink4.jpg")

plt.imshow(img[:,:,::-1])

plt.show()

result = DeepFace.analyze(img,actions=['emotion'])

print(result)