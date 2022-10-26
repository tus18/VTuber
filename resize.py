import glob
import os
import cv2

FM_list = ["男性","女性"]
for FM_name in FM_list:

    #フォルダーを作成している
    os.makedirs("./resize/" + FM_name, exist_ok=True)

    #読み取る画像のフォルダを指定
    in_dir = "./images/" + FM_name + "/*.jpg"
    #出力先のフォルダを指定
    out_dir = "./resize/" + FM_name

    #./assets_kora/drink/のフォルダの画像のディレクトリをすべて配列に格納している
    img_jpg = glob.glob(in_dir)
    #./assets_kora/drink/のファイルを一覧にする

    #画像の個数分繰り返し作業を行う
    for i in range(len(img_jpg)):
        img =cv2.imread(str(img_jpg[i]))
        #64×64サイズにしてる
        img_rot = cv2.resize(img,(64,64))
        #パスを結合している
        fileName = os.path.join(out_dir, str(i) + "_" + str(i) + ".jpg")
        #画像を出力
        cv2.imwrite(str(fileName),img_rot)