import numpy as np
import cv2
import keras
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

model = keras.models.load_model('asl_model')

alphabet = "abcdefghiklmnopqrstuvwxy"
dictionary = {}
for i in range(24):
    dictionary[i] = alphabet[i]

ROI_top = 200
ROI_bottom = 500
ROI_right = 250
ROI_left = 550

cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()

    # filpping the frame to prevent inverted image of captured frame...
    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()

    # ROI from the frame
    roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]
    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresholded = cv2.resize(gray_frame, (28, 28))
    thresholded = np.reshape(thresholded, (1,thresholded.shape[0],thresholded.shape[1],1))
    thresholded = thresholded/255

    pred = model.predict(thresholded)
    cv2.putText(frame_copy, dictionary[np.argmax(pred)], (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,255), 2)

    # Draw ROI on frame_copy
    cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255,128,0), 3)

    # Display the frame with segmented hand
    cv2.putText(frame_copy, "ASL recognition", (10, 20), cv2.FONT_ITALIC, 0.5, (51,255,51), 1)
    cv2.imshow("ASL detection", frame_copy)

    # Close windows with Esc
    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break

# Release the camera and destroy all the windows
cam.release()
cv2.destroyAllWindows()
