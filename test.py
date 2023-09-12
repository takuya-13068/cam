# Import SixDRepNet
from sixdrepnet import SixDRepNet
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time
import os

phase = 0
flg=True
flg2 = False
camera_position_threshold = 50

def img_add_text(img, text, x, y, size, color):
    font_path = 'fonts\arialbd.ttf'
    font_size = size

    font = ImageFont.truetype(font_path, font_size) 
    img = Image.fromarray(img) 
    draw = ImageDraw.Draw(img)
    draw.text((x, y), text, font=font, fill=color)
    
    img = np.array(img)
    return img



# Create model
# Weights are automatically downloaded
model = SixDRepNet()
cap = cv2.VideoCapture(1)
while(cap.isOpened()):
    if flg:
        start = time.perf_counter()
        flg = False
    ret, img = cap.read()
    img = cv2.flip(img,1)
    height, width, channels = img.shape[:3]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    lists = cascade.detectMultiScale(img_gray, minSize=(150, 150))
    if len(lists) == 1:
        for (x,y,w,h) in lists:
            x,y,w,h = x,y,w,h
    else:
        x,y,w,h = 0,0,0,0

    #background write を作る
    cv2.rectangle(img, (int(width*1/20-10), int(height*17/20)), (int(width*19/20+10), int(height-15)), (255,255,255), thickness=-1)
    cv2.rectangle(img, (int(width*1/20-10), int(height*17/20)), (int(width*19/20+10), int(height-15)), (0,0,0))

    #step0: navigation案内
    if phase == 0:
        img = img_add_text(img, 'ナビゲーションを行います。このまましばらくお待ちください。', (int(width* 1/20)), int(height*17/20+10), 16, (0,0,0))
        end = time.perf_counter()

    #step1: 顔認識をする
    if phase == 1:
        img = img_add_text(img, 'Step1: 顔が認識されていません。カメラに近づいてください。', (int(width* 1/20)), int(height*17/20+10), 16, (0,0,255))

    #step2: 頭の位置を調整する
    if phase == 2:
        if len(lists) == 1:
            img = img_add_text(img, 'Step2: 顔の位置が正しくありません。カメラの中心に位置を調節してください。', (int(width* 1/20)), int(height*17/20+10), 16, (0,0,255))
            for (x,y,w,h) in lists:
                x,y,w,h = x,y,w,h

        '''cv2.putText(img,text='center: (x,y)=('+str(int(x+w/2))+','+str(int(y+h/2))+')',org=(10, int(height*1.5/10)),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,color=(255, 255, 255),thickness=2,lineType=cv2.LINE_AA)
        cv2.putText(img,text='error-> x:'+str(x+w/2 - width/2)+'px, y:'+str(y+h/2 - height/2)+'px)',org=(10, int(height*2.5/10)),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,color=(255, 255, 255),thickness=2,lineType=cv2.LINE_AA)'''

    #step2: yaw, roll, pitchをそれぞれ整える。
    if phase == 3:
        if len(lists) == 1:
            for (x,y,w,h) in lists:
                mask_img = img[y:y+h, x:x+w]
        pitch, yaw, roll = model.predict(mask_img)
        img, flg2, output = model.draw_axis(img, yaw, pitch, roll, x,y,w,h, True)

        img = img_add_text(img, 'Step3: 顔の向きや傾きを以下の説明に沿って調整してください。', (int(width* 1/20)), int(height*17/20+10), 16, (0,0,0))

        #print(output)
        #0:ok, 1:minus, 2:plus
        if output[0] != 0: # pitch
            if output[0] == 1:
                mes = '1: 顔を上に向けましょう。'
                img_path = './assets/up.png'
            elif output[0] == 2:
                mes = '1: 顔を下に向けましょう。'
                img_path = './assets/down.png'
        elif output[1] != 0:
            if output[1] == 1:
                mes = '2: 顔を右に向けましょう。'
                img_path = './assets/right.png'
            elif output[1] == 2:
                mes = '2: 顔を左に向けましょう。'
                img_path = './assets/left.png'
        elif output[2] != 0:
            if output[2] == 1:
                mes = '3: 顔を時計回りに傾けましょう。'
                img_path = './assets/rollr.png'
            elif output[2] == 2:
                mes = '3: 顔を反時計回りに傾けましょう。'
                img_path = './assets/rolll.png'
        else:
            mes = '完了'
            img_path=''
        
        img = img_add_text(img, mes, (int(width* 1/20)), int(height*17/20+30), 20, (0,0,255))
        if os.path.exists(img_path):
            img_direct = cv2.imread(img_path)
            img_direct = cv2.resize(img_direct, (100,100))

            img_h, img_w = img_direct.shape[:2]
            img[int(height*13/20):int(height*13/20)+img_h, int(width*4/5-5):int(width*4/5-5)+img_w] = img_direct
        
    if phase == 4:
        img = img_add_text(img, 'キャリブレーションが完了しました。そのままの姿勢でお待ちください。', (int(width* 1/20)), int(height*17/20+10), 16, (0,0,0))
        if len(lists) == 1:
            for (x,y,w,h) in lists:
                mask_img = img[y:y+h, x:x+w]
        pitch, yaw, roll = model.predict(mask_img)
        img, flg2, output = model.draw_axis(img, yaw, pitch, roll, x,y,w,h, False)

    if phase!=0 or end - start > 5:
        if not len(lists) == 1:
            phase = 1
            flg2 = False
        elif not(abs(x+w/2 - width/2) <= camera_position_threshold and abs(y+h/2 - height/2) <= camera_position_threshold):
            phase = 2
            flg2 = False
        elif not flg2:
            phase = 3
        else:
            phase = 4


    # GUIに表示
    cv2.imshow("test_window", img)
    # qキーが押されたら途中終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey(0)

