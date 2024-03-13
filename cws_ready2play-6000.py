import os
import socket
import cv2
import select
from cvzone.PoseModule import PoseDetector

def create_pose_detector():
    return PoseDetector()

def detect_pose(frame, pose_detector):
    frame = pose_detector.findPose(frame)
    landmarks, bbox_info = pose_detector.findPosition(frame)
    return frame, landmarks, bbox_info

def landmarks_to_string(landmarks, frame_height, posList):
    lmString = ''
    for lm in landmarks:
        # lmString += f'{lm[0]},{lm[1]},{frame_height - lm[2]},'
        lmString += f'{lm[0]},{frame_height- lm[1]},{lm[2]},'
    posList.append(lmString)
    # print("posList: ", posList)
    if len(posList) >= 5:
        avgPos = []
        for i in range(len(landmarks)):
            xPos = []
            yPos = []
            zPos = []
            for j in range(len(posList) - 2, len(posList) + 3):
                if j >= 0 and j < len(posList):
                    coords = posList[j].split(',')
                    if i * 3 + 2 < len(coords):
                        xPos.append(float(coords[i * 3]))
                        yPos.append(float(coords[i * 3 + 1]))
                        zPos.append(float(coords[i * 3 + 2]))
            avgX = sum(xPos) / len(xPos)
            avgY = sum(yPos) / len(yPos)
            avgZ = sum(zPos) / len(zPos)
            avgPos.append(f'{avgX},{avgY},{avgZ},')

        avgPosString = ''.join(avgPos)
        print("avgPosString: ", avgPosString)

        return avgPosString
    

def save_motion_data(pose_data_list, file_index):
    directory = "MotionFileFolder"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = f"{directory}/MotionFile{file_index}.txt"
    with open(file_path, 'w') as file:
        file.writelines(f"{item}\n" for item in pose_data_list)        


def increment_file_index(file_index):
    file_path = f"MotionFileFolder/MotionFile{file_index}.txt"
    while os.path.exists(file_path):
        print(f"File {file_path} exists, incrementing file index.")  # 加入日誌輸出
        file_index += 1
        file_path = f"MotionFileFolder/MotionFile{file_index}.txt"  # 更新file_path以反映新的file_index
    print(f"Using file index: {file_index}")  # 輸出最終使用的file_index
    return file_index

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    pose_detector = create_pose_detector()
    posList = [] # Initialize posList here

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('127.0.0.1', 6000))
    client_socket.setblocking(False)

    pose_data_list = []
    record_motion_data = False
    message_from_server = ""

    try:
        while cap.isOpened():
            try: 
                success, frame = cap.read()
                frame = cv2.flip(frame, 1)

                if not success:
                    break

                frame, landmarks, bbox_info = detect_pose(frame, pose_detector)

                ready_to_read, _, _ = select.select([client_socket], [], [], 0)
                if ready_to_read:
                    message_from_server = ""
                    try:
                        message_from_server = client_socket.recv(1024).decode('utf-8')
                    except UnicodeDecodeError:
                        print("Received message could not be decoded to utf-8.")
                        continue
                    except BlockingIOError:
                        continue

                    if message_from_server == 'Server: Ready2Play':
                        print("Server: Ready2Play")
                        client_socket.sendall(('Client: Ready2Play').encode())

                    if message_from_server == 'Server: Record Motion Data':
                        print("Server: Record Motion Data")
                        client_socket.sendall(('Client: Record Motion Data').encode())
                        record_motion_data = True

                    if message_from_server == 'Server: Save Motion Data':
                        print("Server: Save Motion Data")
                        file_index = increment_file_index(1)
                        save_motion_data(pose_data_list, file_index)
                        client_socket.sendall(('Client: Save Motion Data').encode())
                        pose_data_list.clear()
                        record_motion_data = False

                if record_motion_data:
                    if bbox_info:
                        # 之前的版本
                        # landmarks_str = landmarks_to_string(landmarks, frame.shape[0])
                        # pose_data_list.append(landmarks_str)
                        # Modify the call to landmarks_to_string to include posList
                        landmarks_str = landmarks_to_string(landmarks, frame.shape[0], posList)
                        if landmarks_str:
                            pose_data_list.append(f"{landmarks_str}")

                cv2.imshow("Image", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                if message_from_server == 'Server: Exit':
                    client_socket.sendall(('Client: Exit').encode())
                    break
            except Exception as e:
                print(f"An error occurred: {e}")
                continue
    finally:
        cap.release()
        cv2.destroyAllWindows()
        client_socket.close()

if __name__ == "__main__":
    main()