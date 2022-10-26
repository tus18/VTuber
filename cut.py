import cv2
import glob
import os
from PIL import Image

face_cascade_path = './haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

#バウンディングボックス座標データを取得
def Get_Bounding_Box(img_path):
    #画像を読み込む
    img_b = cv2.imread(img_path)
    if img_b is None:
        print("No Object")
        return -1
    #ここでオブジェクトを検出
    #バウンディングボックスもろもろ検出してる
    img_b_gray = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    #バウンディングボックスの座標情報をゲット
    face = face_cascade.detectMultiScale(img_b_gray)
    bbox = []
    for x, y, w, h in face:
        bbox.append([x,y,x+w,y+h])
    print(face)
    #バウンディングボックス座標データを返却
    return bbox

#物体だけを取り出す
def Cut_draw(img_path, bbox_Coordinate, img_number):
    #画像読み込み
    #RGBモードとしてやる RGBAモードだと上手く行かない
    img = Image.open(img_path).convert("RGB")
    #画像を切り取る
    #座標を格納する箱
    item_position_list = []
    #座標を一個ずつ読み取る
    for item_position in bbox_Coordinate:
        for item_coordinate in item_position:
            item_position_list.append(item_coordinate)
        #取得した座標のところだけを抜き取る
        img_crop = img.crop(item_position_list)
        img_crop.convert("RGB")
        #画像出力
        img_crop.save("./images/" + "/cut_image_drink{0}.jpg".format(img_number))
        #座標リストを空にする
        item_position_list.clear()

if __name__ == '__main__':
    #使用する画像のパスを指定する
    img_path = "./img/" + "/*.jpg"
    #１つずつ画像パスを読み取る
    img_jpg = glob.glob(img_path)
    #画像番号
    img_number = 1
    #切り取った画像を保存するフォルダーを作成
    os.makedirs("./images/", exist_ok=True)
    #画像を１つずつ処理する
    for img in img_jpg:
        #バウンディングボックスの座標データを取得&格納
        bbox_Coordinate = Get_Bounding_Box(img)
        if type(bbox_Coordinate) is not int:
            #画像の抜き取り
            Cut_draw(img, bbox_Coordinate, img_number)
            img_number += 1