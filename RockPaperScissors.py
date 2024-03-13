import math
import socket
import time

import cv2
import mediapipe as mp

'''
Use MediaPipe 
&
Detect Hand Gestures (Rock, Paper, and Scissors)
'''

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


# 根據兩點的座標，計算角度
def vector_2d_angle(v1, v2):
    v1_x = v1[0]
    v1_y = v1[1]
    v2_x = v2[0]
    v2_y = v2[1]
    try:
        angle_ = math.degrees(math.acos(
            (v1_x * v2_x + v1_y * v2_y) / (((v1_x ** 2 + v1_y ** 2) ** 0.5) * ((v2_x ** 2 + v2_y ** 2) ** 0.5))))
    except:
        angle_ = 180
    return angle_


# 根據傳入的 21 個節點座標，得到該手指的角度
def hand_angle(hand_):
    angle_list = []
    # thumb 大拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[2][0])), (int(hand_[0][1]) - int(hand_[2][1]))),
        ((int(hand_[3][0]) - int(hand_[4][0])), (int(hand_[3][1]) - int(hand_[4][1])))
    )
    angle_list.append(angle_)
    # index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[6][0])), (int(hand_[0][1]) - int(hand_[6][1]))),
        ((int(hand_[7][0]) - int(hand_[8][0])), (int(hand_[7][1]) - int(hand_[8][1])))
    )
    angle_list.append(angle_)
    # middle 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[10][0])), (int(hand_[0][1]) - int(hand_[10][1]))),
        ((int(hand_[11][0]) - int(hand_[12][0])), (int(hand_[11][1]) - int(hand_[12][1])))
    )
    angle_list.append(angle_)
    # ring 無名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[14][0])), (int(hand_[0][1]) - int(hand_[14][1]))),
        ((int(hand_[15][0]) - int(hand_[16][0])), (int(hand_[15][1]) - int(hand_[16][1])))
    )
    angle_list.append(angle_)
    # pink 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[18][0])), (int(hand_[0][1]) - int(hand_[18][1]))),
        ((int(hand_[19][0]) - int(hand_[20][0])), (int(hand_[19][1]) - int(hand_[20][1])))
    )
    angle_list.append(angle_)
    return angle_list


# 根據手指角度的串列內容，返回對應的手勢名稱
def hand_pos(finger_angle):
    f1 = finger_angle[0]  # 大拇指角度
    f2 = finger_angle[1]  # 食指角度
    f3 = finger_angle[2]  # 中指角度
    f4 = finger_angle[3]  # 無名指角度
    f5 = finger_angle[4]  # 小拇指角度

    # 小於 50 表示手指伸直，大於等於 50 表示手指捲縮
    if f1 < 50 and f2 >= 50 and f3 >= 50 and f4 >= 50 and f5 >= 50:
        return 'good'
    elif f1 >= 50 and f2 >= 50 and f3 < 50 and f4 >= 50 and f5 >= 50:
        return 'no!!!'
    elif f1 < 50 and f2 < 50 and f3 >= 50 and f4 >= 50 and f5 < 50:
        return 'ROCK!'
    elif f1 >= 50 and f2 >= 50 and f3 >= 50 and f4 >= 50 and f5 >= 50:
        return '0'
    elif f1 >= 50 and f2 >= 50 and f3 >= 50 and f4 >= 50 and f5 < 50:
        return 'pink'
    elif f1 >= 50 and f2 < 50 and f3 >= 50 and f4 >= 50 and f5 >= 50:
        return '1'
    elif f1 >= 50 and f2 < 50 and f3 < 50 and f4 >= 50 and f5 >= 50:
        return '2'
    elif f1 >= 50 and f2 >= 50 and f3 < 50 and f4 < 50 and f5 < 50:
        return 'ok'
    elif f1 < 50 and f2 >= 50 and f3 < 50 and f4 < 50 and f5 < 50:
        return 'ok'
    elif f1 >= 50 and f2 < 50 and f3 < 50 and f4 < 50 and f5 > 50:
        return '3'
    elif f1 >= 50 and f2 < 50 and f3 < 50 and f4 < 50 and f5 < 50:
        return '4'
    elif f1 < 50 and f2 < 50 and f3 < 50 and f4 < 50 and f5 < 50:
        return '5'
    elif f1 < 50 and f2 >= 50 and f3 >= 50 and f4 >= 50 and f5 < 50:
        return '6'
    elif f1 < 50 and f2 < 50 and f3 >= 50 and f4 >= 50 and f5 >= 50:
        return '7'
    elif f1 < 50 and f2 < 50 and f3 < 50 and f4 >= 50 and f5 >= 50:
        return '8'
    elif f1 < 50 and f2 < 50 and f3 < 50 and f4 < 50 and f5 >= 50:
        return '9'
    else:
        return ''


cap = cv2.VideoCapture(0)  # 讀取攝影機
fontFace = cv2.FONT_HERSHEY_SIMPLEX  # 印出文字的字型
lineType = cv2.LINE_AA  # 印出文字的邊框

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = '127.0.0.1'  # Localhost IP
port = 7777  # Port to listen on
client_socket.connect((host, port))  # Connect to the server

# mediapipe 啟用偵測手掌
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    start_time_text_eq_zero = None  # 記錄手勢開始時間 (text=0)
    start_time_text_eq_one = None  # 記錄手勢開始時間 (text=1)
    start_time_text_eq_five = None  # 記錄手勢開始時間 (text=5)

    # 計算 print 的次數
    count_print_text_eq_zero = 0
    count_print_text_eq_one = 0
    count_print_text_eq_five = 0

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    w, h = 600, 600  # 影像尺寸
    while True:
        ret, img = cap.read()
        img = cv2.resize(img, (w, h))  # 縮小尺寸，加快處理效率
        if not ret:
            print("Cannot receive frame")
            break
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 轉換成 RGB 色彩
        results = hands.process(img2)  # 偵測手勢
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                finger_points = []  # 記錄手指節點座標的串列
                for i in hand_landmarks.landmark:
                    # 將 21 個節點換算成座標，記錄到 finger_points
                    x = i.x * w
                    y = i.y * h
                    finger_points.append((x, y))
                if finger_points:
                    finger_angle = hand_angle(finger_points)  # 計算手指角度，回傳長度為 5 的串列
                    # print(finger_angle)                     # 印出角度 ( 有需要就開啟註解 )
                    text = hand_pos(finger_angle)  # 取得手勢所回傳的內容
                    # print(text)                             # 印出手勢 ( 有需要就開啟註解 )
                    if text != '':
                        if text == '0':  # 如果是 0 手勢
                            if start_time_text_eq_zero is None:
                                # 記錄手勢開始時間
                                start_time_text_eq_zero = time.time()
                                start_time_text_eq_one = None
                                start_time_text_eq_five = None
                            elif time.time() - start_time_text_eq_zero >= 2:
                                count_print_text_eq_zero += 1
                                count_print_text_eq_one = 0
                                count_print_text_eq_five = 0
                                if count_print_text_eq_zero == 1:
                                    # 印出手勢名稱
                                    print(str(time.time()) + ': ' + text)
                                    client_socket.send("0".encode('utf-8'))
                        elif text == '1':  # 如果是 1 手勢
                            if start_time_text_eq_one is None:
                                # 記錄手勢開始時間
                                start_time_text_eq_one = time.time()
                                start_time_text_eq_zero = None
                                start_time_text_eq_five = None
                            elif time.time() - start_time_text_eq_one >= 2:
                                count_print_text_eq_zero = 0
                                count_print_text_eq_one += 1
                                count_print_text_eq_five = 0
                                if count_print_text_eq_one == 1:
                                    # 印出手勢名稱
                                    print(str(time.time()) + ': ' + text)
                                    client_socket.send("1".encode('utf-8'))
                        elif text == '5':  # 如果是 5 手勢
                            if start_time_text_eq_five is None:
                                # 記錄手勢開始時間
                                start_time_text_eq_five = time.time()
                                start_time_text_eq_zero = None
                                start_time_text_eq_one = None
                            elif time.time() - start_time_text_eq_five >= 2:
                                count_print_text_eq_zero = 0
                                count_print_text_eq_one = 0
                                count_print_text_eq_five += 1
                                if count_print_text_eq_five == 1:
                                    # 印出手勢名稱                                
                                    print(str(time.time()) + ': ' + text)
                                    client_socket.send("5".encode('utf-8'))
                        else:
                            # 重新計時
                            start_time_text_eq_zero = None
                            start_time_text_eq_one = None
                            start_time_text_eq_five = None

                            # 重新計算 print 的次數
                            count_print_text_eq_zero = 0
                            count_print_text_eq_one = 0
                            count_print_text_eq_five = 0
                    cv2.putText(img, text, (30, 120), fontFace, 5, (255, 0, 0), 10, lineType)  # 印出文字
        cv2.imshow('Gesture Recognition', img)
        if cv2.waitKey(5) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
client_socket.close()
