## Put this in the media pipe py torch folder along with the video.webm file
## It shows the bounding box over the video.
import numpy as np
import torch
import cv2

from blazebase import resize_pad, denormalize_detections
from blazeface import BlazeFace
from blazepalm import BlazePalm
from blazeface_landmark import BlazeFaceLandmark
from blazehand_landmark import BlazeHandLandmark

from visualization import draw_detections, draw_landmarks, draw_roi, HAND_CONNECTIONS

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)



face_detector = BlazeFace().to(gpu)
face_detector.load_weights("blazeface.pth")
face_detector.load_anchors("anchors_face.npy")

face_regressor = BlazeFaceLandmark().to(gpu)
face_regressor.load_weights("blazeface_landmark.pth")


WINDOW='test'
cv2.namedWindow(WINDOW)
capture = cv2.VideoCapture('video.webm')
success,image = capture.read()

if capture.isOpened():
    hasFrame, frame = capture.read()
    frame_ct = 0
else:
    hasFrame = False

while hasFrame:
    frame_ct +=1

    frame = np.ascontiguousarray(frame[:,::-1,::-1])

    img1, img2, scale, pad = resize_pad(frame)

    normalized_face_detections = face_detector.predict_on_image(img2)

    face_detections = denormalize_detections(normalized_face_detections, scale, pad)


    xc, yc, scale, theta = face_detector.detection2roi(face_detections)
    img, affine, box = face_regressor.extract_roi(frame, xc, yc, theta, scale)
    flags, normalized_landmarks = face_regressor(img)
    landmarks = face_regressor.denormalize_landmarks(normalized_landmarks, affine)



    for i in range(len(flags)):
        landmark, flag = landmarks[i], flags[i]
        if flag>.5:
            draw_landmarks(frame, landmark[:,:2], size=1)


    draw_roi(frame, box)
    draw_detections(frame, face_detections)

    cv2.imshow(WINDOW, frame[:,:,::-1])
    # cv2.imwrite('sample/%04d.jpg'%frame_ct, frame[:,:,::-1])

    hasFrame, frame = capture.read()
    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()
