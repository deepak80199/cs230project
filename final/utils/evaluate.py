# evaluater

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


def evaluate_pose(user_df, pose, vid_path):
    """
    Evaluate the user pose against instruction pose

    Parameters:
        user_df (Pandas.dataframe): Dataframe consisting of 8 angles for each frame
        pose (int): Classification of user pose
        vid_path(string): Absolute to user vid path

    Returns:
        None
    """

    ins_df = pd.read_csv('ins_pose/' + str(pose) + '.csv')

    angle_diff_df = get_angle_diff(user_df, ins_df)

    angle_diff_df_copy = angle_diff_df
    angle_diff_df_color = angle_diff_df_copy.applymap(angleColor)

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(vid_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    df_user = user_df
    df_angle = angle_diff_df
    df_color = angle_diff_df_color
    x = 0

    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        i = 0
        while cap.isOpened():
            df = list(df_user.iloc[x].values)
            angle_name_lst = df_angle.iloc[x]
            line_color_lst = df_color.iloc[x]

            ret, frame = cap.read()

            if not ret:
                break

            # Recolor image to RGB
            image1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image1.flags.writeable = False

            # Make detection
            results = pose.process(image1)

            # Recolor back to BGR
            image1.flags.writeable = True
            image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                # Get coordinates
                landmarks = results.pose_landmarks.landmark
                setOfJoints = mp_pose.PoseLandmark

                leftWrist = [landmarks[setOfJoints.LEFT_WRIST].x, landmarks[setOfJoints.LEFT_WRIST].y]
                leftElbow = [landmarks[setOfJoints.LEFT_ELBOW].x, landmarks[setOfJoints.LEFT_ELBOW].y]
                leftShoulder = [landmarks[setOfJoints.LEFT_SHOULDER].x, landmarks[setOfJoints.LEFT_SHOULDER].y]
                leftHip = [landmarks[setOfJoints.LEFT_HIP].x, landmarks[setOfJoints.LEFT_HIP].y]
                leftKnee = [landmarks[setOfJoints.LEFT_KNEE].x, landmarks[setOfJoints.LEFT_KNEE].y]
                leftAnkle = [landmarks[setOfJoints.LEFT_ANKLE].x, landmarks[setOfJoints.LEFT_ANKLE].y]
                leftEye = [landmarks[setOfJoints.LEFT_EYE].x, landmarks[setOfJoints.LEFT_EYE].y]
                leftEyeinner = [landmarks[setOfJoints.LEFT_EYE_INNER].x, landmarks[setOfJoints.LEFT_EYE_INNER].y]
                leftEar = [landmarks[setOfJoints.LEFT_EAR].x, landmarks[setOfJoints.LEFT_EAR].y]
                leftPinky = [landmarks[setOfJoints.LEFT_PINKY].x, landmarks[setOfJoints.LEFT_PINKY].y]
                leftThumb = [landmarks[setOfJoints.LEFT_THUMB].x, landmarks[setOfJoints.LEFT_THUMB].y]
                leftHeel = [landmarks[setOfJoints.LEFT_HEEL].x, landmarks[setOfJoints.LEFT_HEEL].y]
                leftFootindex = [landmarks[setOfJoints.LEFT_FOOT_INDEX].x, landmarks[setOfJoints.LEFT_FOOT_INDEX].y]

                nose = [landmarks[setOfJoints.NOSE].x, landmarks[setOfJoints.NOSE].y]

                rightWrist = [landmarks[setOfJoints.RIGHT_WRIST].x, landmarks[setOfJoints.RIGHT_WRIST].y]
                rightElbow = [landmarks[setOfJoints.RIGHT_ELBOW].x, landmarks[setOfJoints.RIGHT_ELBOW].y]
                rightShoulder = [landmarks[setOfJoints.RIGHT_SHOULDER].x, landmarks[setOfJoints.RIGHT_SHOULDER].y]
                rightHip = [landmarks[setOfJoints.RIGHT_HIP].x, landmarks[setOfJoints.RIGHT_HIP].y]
                rightKnee = [landmarks[setOfJoints.RIGHT_KNEE].x, landmarks[setOfJoints.RIGHT_KNEE].y]
                rightAnkle = [landmarks[setOfJoints.RIGHT_ANKLE].x, landmarks[setOfJoints.RIGHT_ANKLE].y]
                rightEye = [landmarks[setOfJoints.RIGHT_EYE].x, landmarks[setOfJoints.RIGHT_EYE].y]
                rightEyeinner = [landmarks[setOfJoints.RIGHT_EYE_INNER].x, landmarks[setOfJoints.RIGHT_EYE_INNER].y]
                rightEar = [landmarks[setOfJoints.RIGHT_EAR].x, landmarks[setOfJoints.RIGHT_EAR].y]
                rightPinky = [landmarks[setOfJoints.RIGHT_PINKY].x, landmarks[setOfJoints.RIGHT_PINKY].y]
                rightThumb = [landmarks[setOfJoints.RIGHT_THUMB].x, landmarks[setOfJoints.RIGHT_THUMB].y]
                rightHeel = [landmarks[setOfJoints.RIGHT_HEEL].x, landmarks[setOfJoints.RIGHT_HEEL].y]
                rightFootindex = [landmarks[setOfJoints.RIGHT_FOOT_INDEX].x, landmarks[setOfJoints.RIGHT_FOOT_INDEX].y]

                angleDict = {'leftWES': (leftWrist, leftElbow, leftShoulder),
                             'leftESH': (leftElbow, leftShoulder, leftHip),
                             'leftSHK': (leftShoulder, leftHip, leftKnee),
                             'leftHKA': (leftHip, leftKnee, leftAnkle),
                             'rightWES': (rightWrist, rightElbow, rightShoulder),
                             'rightESH': (rightElbow, rightShoulder, rightHip),
                             'rightSHK': (rightShoulder, rightHip, rightKnee),
                             'rightHKA': (rightHip, rightKnee, rightAnkle)}

                # Blank Frame for Skeleton
                image = np.zeros((int(height), int(width), 3), dtype='uint8')
                # image.fill(255)

                # Orange : (255, 69, 0)
                # Red : (255,0,0)

                # Visualize angle
                # 1.Elbow joint
                # Left Elbow
                # 'leftWES': (leftWrist, leftElbow, leftShoulder),

                cv2.putText(image, str(angle_name_lst['leftWES']),
                            tuple(np.multiply(leftElbow, [width + 20, height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.line(image, tuple(np.multiply(leftShoulder, [width, height]).astype(int)),
                         tuple(np.multiply(leftElbow, [width, height]).astype(int)), (0, 255, 0), thickness=5)
                cv2.line(image, tuple(np.multiply(leftWrist, [width, height]).astype(int)),
                         tuple(np.multiply(leftElbow, [width, height]).astype(int)), line_color_lst['leftWES'],
                         thickness=5)

                # Right Elbow
                # 'rightWES': (rightWrist, rightElbow, rightShoulder)

                cv2.putText(image, str(angle_name_lst['rightWES']),
                            tuple(np.multiply(rightElbow, [width - 330, height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.line(image, tuple(np.multiply(rightShoulder, [width, height]).astype(int)),
                         tuple(np.multiply(rightElbow, [width, height]).astype(int)), (0, 255, 0), thickness=5)
                cv2.line(image, tuple(np.multiply(rightWrist, [width, height]).astype(int)),
                         tuple(np.multiply(rightElbow, [width, height]).astype(int)), line_color_lst['rightWES'],
                         thickness=5)

                # 2.Shoulder joint
                # Left Shoulder
                # 'leftESH': (leftElbow, leftShoulder, leftHip),

                cv2.putText(image, str(angle_name_lst['leftESH']),
                            tuple(np.multiply(leftShoulder, [width + 20, height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.line(image, tuple(np.multiply(leftShoulder, [width, height]).astype(int)),
                         tuple(np.multiply(leftElbow, [width, height]).astype(int)), line_color_lst['leftESH'],
                         thickness=5)
                cv2.line(image, tuple(np.multiply(leftHip, [width, height]).astype(int)),
                         tuple(np.multiply(leftShoulder, [width, height]).astype(int)), (0, 255, 0), thickness=5)

                # Right Shoulder
                # 'rightESH': (rightElbow, rightShoulder, rightHip),

                cv2.putText(image, str(angle_name_lst['rightESH']),
                            tuple(np.multiply(rightShoulder, [width - 350, height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.line(image, tuple(np.multiply(rightShoulder, [width, height]).astype(int)),
                         tuple(np.multiply(rightElbow, [width, height]).astype(int)), line_color_lst['rightESH'],
                         thickness=5)
                cv2.line(image, tuple(np.multiply(rightHip, [width, height]).astype(int)),
                         tuple(np.multiply(rightShoulder, [width, height]).astype(int)), (0, 255, 0), thickness=5)

                # 3.Hip joint
                # Left Hip
                # 'leftSHK': (leftShoulder, leftHip, leftKnee),

                cv2.putText(image, str(angle_name_lst['leftSHK']),
                            tuple(np.multiply(leftHip, [width + 20, height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.line(image, tuple(np.multiply(leftShoulder, [width, height]).astype(int)),
                         tuple(np.multiply(leftHip, [width, height]).astype(int)), line_color_lst['leftSHK'],
                         thickness=5)
                cv2.line(image, tuple(np.multiply(leftHip, [width, height]).astype(int)),
                         tuple(np.multiply(leftKnee, [width, height]).astype(int)), (0, 255, 0), thickness=5)

                # Right Hip
                # 'rightSHK': (rightShoulder, rightHip, rightKnee),

                cv2.putText(image, str(angle_name_lst['rightSHK']),
                            tuple(np.multiply(rightHip, [width - 320, height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.line(image, tuple(np.multiply(rightShoulder, [width, height]).astype(int)),
                         tuple(np.multiply(rightHip, [width, height]).astype(int)), line_color_lst['rightSHK'],
                         thickness=5)
                cv2.line(image, tuple(np.multiply(rightHip, [width, height]).astype(int)),
                         tuple(np.multiply(rightKnee, [width, height]).astype(int)), (0, 255, 0), thickness=5)

                # 4.Knee joint
                # Left Knee
                # 'leftHKA': (leftHip, leftKnee, leftAnkle),
                cv2.putText(image, str(angle_name_lst['leftHKA']),
                            tuple(np.multiply(leftKnee, [width + 20, height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.line(image, tuple(np.multiply(leftKnee, [width, height]).astype(int)),
                         tuple(np.multiply(leftHip, [width, height]).astype(int)), line_color_lst['leftHKA'],
                         thickness=5)
                cv2.line(image, tuple(np.multiply(leftKnee, [width, height]).astype(int)),
                         tuple(np.multiply(leftAnkle, [width, height]).astype(int)), (0, 255, 0), thickness=5)

                # Right Knee
                # 'rightHKA': (rightHip, rightKnee, rightAnkle),

                cv2.putText(image, str(angle_name_lst['rightHKA']),
                            tuple(np.multiply(rightKnee, [width - 420, height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.line(image, tuple(np.multiply(rightKnee, [width, height]).astype(int)),
                         tuple(np.multiply(rightHip, [width, height]).astype(int)), line_color_lst['rightHKA'],
                         thickness=5)
                cv2.line(image, tuple(np.multiply(rightAnkle, [width, height]).astype(int)),
                         tuple(np.multiply(rightKnee, [width, height]).astype(int)), (0, 255, 0), thickness=5)

                # connecting wrist to index and thumb
                # right
                cv2.line(image, tuple(np.multiply(rightWrist, [width, height]).astype(int)),
                         tuple(np.multiply(rightPinky, [width, height]).astype(int)), (0, 255, 0), thickness=5)
                cv2.line(image, tuple(np.multiply(rightWrist, [width, height]).astype(int)),
                         tuple(np.multiply(rightThumb, [width, height]).astype(int)), (0, 255, 0), thickness=5)
                # left
                cv2.line(image, tuple(np.multiply(leftWrist, [width, height]).astype(int)),
                         tuple(np.multiply(leftPinky, [width, height]).astype(int)), (0, 255, 0), thickness=5)
                cv2.line(image, tuple(np.multiply(leftWrist, [width, height]).astype(int)),
                         tuple(np.multiply(leftThumb, [width, height]).astype(int)), (0, 255, 0), thickness=5)

                # connecting ankle to index and heel
                # right
                cv2.line(image, tuple(np.multiply(rightAnkle, [width, height]).astype(int)),
                         tuple(np.multiply(rightFootindex, [width, height]).astype(int)), (0, 255, 0), thickness=5)
                cv2.line(image, tuple(np.multiply(rightAnkle, [width, height]).astype(int)),
                         tuple(np.multiply(rightHeel, [width, height]).astype(int)), (0, 255, 0), thickness=5)
                # left
                cv2.line(image, tuple(np.multiply(leftAnkle, [width, height]).astype(int)),
                         tuple(np.multiply(leftFootindex, [width, height]).astype(int)), (0, 255, 0), thickness=5)
                cv2.line(image, tuple(np.multiply(leftAnkle, [width, height]).astype(int)),
                         tuple(np.multiply(leftFootindex, [width, height]).astype(int)), (0, 255, 0), thickness=5)

                # Connecting Shoulders
                cv2.line(image, tuple(np.multiply(rightShoulder, [width, height]).astype(int)),
                         tuple(np.multiply(leftShoulder, [width, height]).astype(int)), (0, 255, 0), thickness=5)

                # Shoulder connection midpoint calculation
                x1, y1 = tuple(np.multiply(rightShoulder, [width, height]).astype(int))
                x2, y2 = tuple(np.multiply(leftShoulder, [width, height]).astype(int))
                x_m_point = (x1 + x2) // 2
                y_m_point = (y1 + y2) // 2

                # connecting nose and shoulder midpoint
                cv2.line(image, (x_m_point, y_m_point),
                         tuple(np.multiply(nose, [width, height]).astype(int)), (0, 255, 0), thickness=5)

                # Connecting Hip
                cv2.line(image, tuple(np.multiply(rightHip, [width, height]).astype(int)),
                         tuple(np.multiply(leftHip, [width, height]).astype(int)), (0, 255, 0), thickness=5)

                # Connecting right eye to nose
                cv2.line(image, tuple(np.multiply(rightEyeinner, [width, height]).astype(int)),
                         tuple(np.multiply(rightEye, [width, height]).astype(int)), (0, 255, 0), thickness=5)
                cv2.line(image, tuple(np.multiply(rightEyeinner, [width, height]).astype(int)),
                         tuple(np.multiply(nose, [width, height]).astype(int)), (0, 255, 0), thickness=5)

                # Connecting left eye to nose
                cv2.line(image, tuple(np.multiply(leftEyeinner, [width, height]).astype(int)),
                         tuple(np.multiply(leftEye, [width, height]).astype(int)), (0, 255, 0), thickness=5)
                cv2.line(image, tuple(np.multiply(leftEyeinner, [width, height]).astype(int)),
                         tuple(np.multiply(nose, [width, height]).astype(int)), (0, 255, 0), thickness=5)

                # Connecting left eye to left ear
                cv2.line(image, tuple(np.multiply(leftEye, [width, height]).astype(int)),
                         tuple(np.multiply(leftEar, [width, height]).astype(int)), (0, 255, 0), thickness=5)

                # Connecting right eye to right ear
                cv2.line(image, tuple(np.multiply(rightEye, [width, height]).astype(int)),
                         tuple(np.multiply(rightEar, [width, height]).astype(int)), (0, 255, 0), thickness=5)

            except:
                pass

            # Display User Feed
            cv2.imshow('Saved User Feed', image1)
            # Display Skeleton Frame
            cv2.imshow('Skeleton', image)
            # Capture the frames
            cv2.imwrite(f'{"video/"}frame{i}.jpg', image)
            # overlay
            tr = 0.3  # transparency between 0-1, show camera if 0
            frame = ((1 - tr) * image1.astype(np.float) + tr * image.astype(np.float)).astype(np.uint8)
            # out.write(cv2.flip(frame,0))
            cv2.imshow('Transparent result', frame)
            cv2.imwrite(f'{"vid/"}frame{i}.png', frame)
            i = i + 1

            x = x + 1

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def create_vid():
    img = []
    for i in range(0, 500):
        img.append(cv2.imread('vid/frame' + str(i) + '.png'))

    height, width, layers = img[1].shape
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video = cv2.VideoWriter('videoyoga.mp4', fourcc, 20, (width, height))

    for j in range(0, 500):
        video.write(img[j])

    cv2.destroyAllWindows()
    video.release()


def get_angle_diff(df_user, df_ins):
    """
    Calculate the difference between user and instructor pose using DTW

    Parameters:
        df_user (Pandas.dataframe): Dataframe consisting of 8 angles for each frame of user
        df_ins (Pandas.dataframe): Dataframe consisting of 8 angles for each frame of instructor

    Returns:
        angle_diff_df (Pandas.dataframe): Difference of user and instructor pose
    """
    angles = ['leftWES', 'leftESH', 'leftSHK', 'leftHKA', 'rightWES', 'rightESH', 'rightSHK', 'rightHKA']
    angle_dtw_dict = {}
    length = len(df_user['leftWES'].values)
    angle_diff_dict = {'leftWES': [0] * length, 'leftESH': [0] * length, 'leftSHK': [0] * length,
                       'leftHKA': [0] * length, 'rightWES': [0] * length, 'rightESH': [0] * length,
                       'rightSHK': [0] * length, 'rightHKA': [0] * length}
    for i in angles:
        ls = list(df_user[i].values)
        ls2 = list(df_ins[i].values)
        distance, path = fastdtw(ls, ls2)
        angle_dtw_dict[i] = {'distance': distance, 'path': path}

        for n, t in enumerate(angle_dtw_dict[i]['path']):
            angle_diff_dict[i][t[0]] = (abs(ls[t[0]] - ls2[t[1]]))

        angle_diff_df = pd.DataFrame(angle_diff_dict)

    return angle_diff_df


def angleColor(angle):
    """
    Calculate the color based on the difference

    Parameters:
        angle (int): angle

    Returns:
        color (tuple(int,int,int)): tuple of RGB values to representing the color
    """
    if angle <= 25:
        return (0, 255, 0)
    elif angle <= 50:
        return (0, 0, 255)
    else:
        return (255, 0, 0)


def calculateAngle(x):
    """
    Calculate the angle based on the given 3 points

    Parameters:
        x (list(list(int))): list of 3 x,y coordinate

    Returns:
        angle (int): angle formed by the 3 coordinated
    """
    a = np.array(x[0])
    b = np.array(x[1])
    c = np.array(x[2])

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return int(angle)
