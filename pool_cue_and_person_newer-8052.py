import os
import socket
import select
import cv2
import numpy as np
import argparse
from ultralytics import YOLO
import time
from fastdtw import fastdtw

# Constants
SERVER_ADDRESS = ('127.0.0.1', 8052)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
YOLO_SEGMENTATION_MODEL_PATH = 'C:\\Users\\admin\\Desktop\\Python\\BallSpeed\\yolov8\\runs\\segment\\train24\\weights\\best.pt'
YOLO_POSE_MODEL_PATH = 'yolov8n-pose.pt'

# Helper functions
def calculate_angle(keypoints, shoulder_index, elbow_index, wrist_index):
    # Calculate the angle between the shoulder, elbow, and wrist keypoints
    shoulder_x, shoulder_y = keypoints[0, shoulder_index]
    elbow_x, elbow_y = keypoints[0, elbow_index]
    wrist_x, wrist_y = keypoints[0, wrist_index]

    vector_shoulder_to_elbow = elbow_x.item() - shoulder_x.item()
    vector_elbow_to_wrist = wrist_x.item() - elbow_x.item()

    vector_shoulder_to_elbow = np.array([elbow_x.item() - shoulder_x.item(), elbow_y.item() - shoulder_y.item()])
    vector_elbow_to_wrist = np.array([wrist_x.item() - elbow_x.item(), wrist_y.item() - elbow_y.item()])

    dot_product = np.dot(vector_shoulder_to_elbow, vector_elbow_to_wrist)
    norm_product = np.linalg.norm(vector_shoulder_to_elbow) * np.linalg.norm(vector_elbow_to_wrist)

    if norm_product == 0:
        return 0

    cosine_of_angle = dot_product / norm_product
    angle_in_radians = np.arccos(np.clip(cosine_of_angle, -1.0, 1.0))
    angle_in_degrees = np.degrees(angle_in_radians)

    return angle_in_degrees

def read_data(file_name):
    with open(file_name, 'r', encoding='UTF-8') as file:
        return np.array([float(num) for num in file.read().split()], dtype=float)

def save_motion_angle(angle_data_list):
    if not angle_data_list:
        return
    file_index = 1
    while os.path.exists(f"MotionAngleFileFolder/MotionAngleFile{file_index}.txt"):
        file_index += 1
    with open(f"MotionAngleFileFolder/MotionAngleFile{file_index}.txt", 'w') as file:
        for item in angle_data_list:
            file.write(f"{item}\n")

    standard_path = r'C:\Users\admin\Desktop\Python\dtw\txt_file\standard_motion_angle.txt'
    array_standard_motion_angle = read_data(standard_path)

    strike_path = f'C:\\Users\\admin\\Desktop\\Python\\MotionAngleFileFolder\\MotionAngleFile{file_index}.txt'
    array_strike_motion_angle = read_data(strike_path)
    
    distance, warp_path = fastdtw(array_standard_motion_angle, array_strike_motion_angle)
    score_folder_path = 'C:/Users/admin/Desktop/Python/dtw/DTW-Score'

    existing_files = os.listdir(score_folder_path)
    file_count = len([f for f in existing_files if f.startswith('dtw_score')])
    new_file_name = f'dtw_score{file_count + 1}.txt'
    score_file_path = os.path.join(score_folder_path, new_file_name)

    with open(score_file_path, 'w') as score_file:
        score_file.write(f"{int(distance)}")
    
    print(f"DTW Distance: {distance}\nDTW Path: {warp_path}")

def save_spin_type(spin_type_list, file_index):
    if spin_type_list == []:
        return
    file_index = 1
    while os.path.exists(f"SpinTypeFileFolder/SpinTypeFile{file_index}.txt"):
        file_index += 1
    with open(f"SpinTypeFileFolder/SpinTypeFile{file_index}.txt", 'w') as file:
        for item in spin_type_list:
            file.write(f"{item}\n")

def main(hand):
    # Main function to run the YOLO models and process video frames
    model = YOLO(YOLO_SEGMENTATION_MODEL_PATH)
    model_pose = YOLO(YOLO_POSE_MODEL_PATH)
    
    # Block of code for setting up the camera
    cap = cv2.VideoCapture(1)
    cap.set(3, CAMERA_WIDTH)
    cap.set(4, CAMERA_HEIGHT)

    # Block of code for setting up the client socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(SERVER_ADDRESS)
    client_socket.setblocking(False)
    
    # Block of code for setting up variables    
    is_ready2play_from_server = False
    is_gripping_correctly = False
    temp_angle_data_list = []
    temp_spin_type_list = []
    cue_stick_X_coordinates_list = []
    cue_stick_Y_coordinates_list = []
    hand_X_coordinate = None
    count_play_times = 0
    # 初始化變量來存儲上一次添加數據的時間
    last_append_time = 0 
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame = cv2.flip(frame, 1)
            # Process frame with YOLO models
            # Block of code for processing the segmentation results and pose results
            seg_results = model.predict(source=frame, show=False, save=False, show_labels=True, show_conf=True, conf=0.8,
                                    save_txt=False, save_crop=False, line_width=2, box=True, classes=0, boxes=False)
            pose_results = model_pose(source=frame, conf=0.8)                
            
            # Block of code to receive data from the server 
            ready_to_read, _, _ = select.select([client_socket], [], [], 0)
            if ready_to_read:
                try:
                    data_from_server = client_socket.recv(1024).decode('utf-8')
                except UnicodeDecodeError:
                    print("Error decoding data from server")
                    continue
                except BlockingIOError:
                    print("BlockingIOError")
                    continue
                if data_from_server == 'Server: CheckGripPosition':
                    print(f"Server: CheckGripPosition")                    
                    is_ready2play_from_server = True
                if data_from_server == 'Server: SaveMotionData':
                    print(f"Server: SaveMotionData")
                    save_motion_angle(temp_angle_data_list)
                    save_spin_type(temp_spin_type_list, count_play_times)
                    is_ready2play_from_server = False
                    is_gripping_correctly = False
                    temp_angle_data_list.clear()
                    temp_spin_type_list.clear()
                    cue_stick_X_coordinates_list.clear()
                    cue_stick_Y_coordinates_list.clear()
                    hand_X_coordinate = None

            # Block of code to process the segmentation results
            for result_mask in seg_results:
                if result_mask.masks is not None:
                    segments = result_mask.masks.xy
                    points_x = segments[0][:, 0]
                    points_y = segments[0][:, 1]
                    cue_stick_X_coordinates_list = points_x.tolist()
                    cue_stick_Y_coordinates_list = points_y.tolist()

                    if is_gripping_correctly == True:
                        min_x = min(cue_stick_X_coordinates_list)
                        min_x_index = cue_stick_X_coordinates_list.index(min_x)
                        max_x = max(cue_stick_X_coordinates_list)
                        max_x_index = cue_stick_X_coordinates_list.index(max_x)

                        # Camera shot from the left
                        if str(args.hand) == 'left':
                            cue_stick_Y_front = cue_stick_Y_coordinates_list[min_x_index]
                            cue_stick_Y_back = cue_stick_Y_coordinates_list[max_x_index]
                        if str(args.hand) == 'right':
                            cue_stick_Y_front = cue_stick_Y_coordinates_list[max_x_index]
                            cue_stick_Y_back = cue_stick_Y_coordinates_list[min_x_index]

                        # Top Spin, No Spin & Back Spin
                        if cue_stick_Y_back > cue_stick_Y_front:
                            temp_spin_type_list.append("Top Spin")
                        elif cue_stick_Y_back == cue_stick_Y_front:
                            temp_spin_type_list.append("No Spin")
                        elif cue_stick_Y_back < cue_stick_Y_front:
                            temp_spin_type_list.append("Back Spin")

            # Block of code to determine if the user is gripping the cue correctly
            for result_pose in pose_results:
                keypoints = result_pose.keypoints.xy
                # check if key points array is not empty
                if keypoints.shape[1] > 0:
                    # 獲取當前時間
                    current_time = time.time()
                    if str(args.hand) == 'left':
                        angle_data = calculate_angle(keypoints, 6, 8, 10)
                        print(f"is_gripping_correctly: {is_gripping_correctly}")
                        # 如果正在正確地握住球桿並且距離上一次添加數據的時間大於等於 3 秒
                        if is_gripping_correctly == True and current_time - last_append_time >= 3:
                            print(f"angle_data: {angle_data}")
                            temp_angle_data_list.append(angle_data)
                            # 更新上一次添加數據的時間
                            last_append_time = current_time

                        left_wrist_x, _ = keypoints[0, 10]
                        hand_X_coordinate = left_wrist_x.item()
                    if str(args.hand) == 'right':
                        angle_data = calculate_angle(keypoints, 5, 7, 9)
                        print(f"is_gripping_correctly: {is_gripping_correctly}")
                        # 如果正在正確地握住球桿並且距離上一次添加數據的時間大於等於 3 秒
                        if is_gripping_correctly == True and current_time - last_append_time >= 1:
                            print(f"angle_data: {angle_data}")
                            temp_angle_data_list.append(angle_data)
                            # 更新上一次添加數據的時間
                            last_append_time = current_time

                        right_wrist_x, _ = keypoints[0, 9]                        
                        hand_X_coordinate = right_wrist_x.item()
            
            if cue_stick_X_coordinates_list and hand_X_coordinate and is_ready2play_from_server:
                hand_closer_to_min = abs(hand_X_coordinate - min(cue_stick_X_coordinates_list)) < abs(hand_X_coordinate - max(cue_stick_X_coordinates_list))
                hand_closer_to_max = abs(hand_X_coordinate - max(cue_stick_X_coordinates_list)) < abs(hand_X_coordinate - min(cue_stick_X_coordinates_list))

                if str(args.hand) == 'left':
                    if hand_closer_to_min:
                        is_gripping_correctly = True
                        is_ready2play_from_server = False
                        client_socket.send("Client: isGrippingCorrectly".encode())
                elif str(args.hand) == 'right':
                    if hand_closer_to_max:
                        is_gripping_correctly = True
                        is_ready2play_from_server = False                        
                        client_socket.send("Client: isGrippingCorrectly".encode())

                # clear
                hand_closer_to_min = None
                hand_closer_to_max = None
                hand_X_coordinate = None
                cue_stick_X_coordinates_list = []
                cue_stick_Y_coordinates_list = []

            # Block of code for plotting the results
            annotated_frame_seg = seg_results[0].plot()
            annotated_frame_pose = pose_results[0].plot()            

            # Show the frame with the plotted results
            cv2.imshow("YOLOv8 Inference Seg", annotated_frame_seg)
            cv2.imshow("YOLOv8 Inference Pose", annotated_frame_pose)
            
            # If 'q' is pressed, break the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Exception: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        client_socket.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pool Cue Analysis Tool')
    parser.add_argument('--hand', type=str, help='the hand to use', required=True)
    args = parser.parse_args()
    main(args.hand)
