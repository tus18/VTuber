import cv2 as cv
import dlib
from imutils import face_utils
from scipy.spatial import distance
import numpy as np
from PIL import Image

img1="D:\ドキュメント\openCV_program\emotion\main.png"
img2="D:\ドキュメント\openCV_program\emotion\close.png"
img3="D:\ドキュメント\openCV_program\emotion\happy.png"
img4="D:\ドキュメント\openCV_program\emotion\ang.png"

cv_img1 = cv.imread(img1,cv.IMREAD_UNCHANGED)
cv_img2 = cv.imread(img2,cv.IMREAD_UNCHANGED)
cv_img3 = cv.imread(img3,cv.IMREAD_UNCHANGED)
cv_img4 = cv.imread(img4,cv.IMREAD_UNCHANGED)

def overlayImage(src, location, size, emotion):
    if emotion == "happy":
        overlay = cv_img3
    else:
        overlay = cv_img1

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