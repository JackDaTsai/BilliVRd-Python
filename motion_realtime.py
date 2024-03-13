import socket
import cv2
from cvzone.PoseModule import PoseDetector

# Initialize video capture with default camera
video_capture = cv2.VideoCapture(0)

# Create a pose detector instance
pose_detector = PoseDetector()

# List to store pose data
pose_data_list = []

# Setup UDP server Calculate_and_Socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ("127.0.0.1", 5054)

try:
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # If frame capture failed, exit loop
        if not ret:
            break

        # Mirror the frame
        frame = cv2.flip(frame, 1)

        # Detect and draw pose
        frame = pose_detector.findPose(frame)
        # Extract pose landmarks
        landmarks, bbox_info = pose_detector.findPosition(frame)

        # If landmarks are detected, process and send them
        if bbox_info:
            # Convert landmarks to string
            landmarks_str = ','.join(
                [','.join(map(str, [lm[0], -lm[1], frame.shape[0] - lm[2]])) for lm in landmarks]
            )
            # Append to pose data list
            pose_data_list.append(landmarks_str)

            # Send pose data to client
            server_socket.sendto(f"{landmarks_str}\n".encode(), server_address)

        # Display the resulting frame
        cv2.imshow('Pose Detection', frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # When everything done, release the capture and close windows
    video_capture.release()
    cv2.destroyAllWindows()
    # Close the server Calculate_and_Socket
    server_socket.close()
