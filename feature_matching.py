import cv2
import numpy as np
import random

cap1 = cv2.VideoCapture('video1.MOV')
cap2 = cv2.VideoCapture('Video2.MOV')

orb = cv2.ORB_create(nfeatures=1000)

FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=12, key_size=20, multi_probe_level=2)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

output_width = 1280
output_height = 720

out = cv2.VideoWriter('output_features.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 60, (output_width, output_height))

while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    good_matches = []
    if des1 is not None and des2 is not None:
        matches = flann.knnMatch(des1, des2, k=2)

        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        good_matches = sorted(good_matches, key=lambda x: x.distance)

        combined_frame = np.hstack((frame1, frame2))

        for match in good_matches[:30]:
            pt1 = tuple(np.int32(kp1[match.queryIdx].pt))
            pt2 = tuple(np.int32(kp2[match.trainIdx].pt) + np.array([frame1.shape[1], 0]))
            color = (random.randint(100, 255), random.randint(50, 255), random.randint(50, 255))
            cv2.line(combined_frame, pt1, pt2, color, 5)

        scale_w = output_width / combined_frame.shape[1]
        scale_h = output_height / combined_frame.shape[0]
        scale = min(scale_w, scale_h)

        new_w = int(combined_frame.shape[1] * scale)
        new_h = int(combined_frame.shape[0] * scale)
        resized_frame = cv2.resize(combined_frame, (new_w, new_h))

        top_pad = (output_height - new_h) // 2
        bottom_pad = output_height - new_h - top_pad
        left_pad = (output_width - new_w) // 2
        right_pad = output_width - new_w - left_pad

        final_frame = cv2.copyMakeBorder(resized_frame, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))

        out.write(final_frame)
        cv2.imshow('Feature Matching (Resized)', final_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap1.release()
cap2.release()
out.release()
cv2.destroyAllWindows()