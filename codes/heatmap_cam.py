from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from scipy.ndimage import zoom
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

from zmq import NULL


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    conv_output = maskNet.get_layer("out_relu").output
    pred_output = maskNet.get_layer("dense_1").output
    maskNet = Model(maskNet.input, outputs=[conv_output, pred_output])

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            # back_face = face
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")

        frame = cv2.resize(frame, (224, 224))
        frame = np.asarray([frame])

        conv, preds = maskNet.predict(faces, batch_size=32)
        # print(conv)

        target = np.argmax(preds, axis=1).squeeze()
        w, b = maskNet.get_layer("dense").weights
        weights = w[:, target].numpy()
        heatmap = conv.squeeze() @ weights
        scale = 224 / 7
        conv = zoom(heatmap, zoom=(scale, scale))

        conv = ((conv - np.min(conv)) / np.ptp(conv) * 256).astype("uint8")
        heatmap = cv2.applyColorMap(conv, cv2.COLORMAP_HOT)

        # cmap = plt.get_cmap('jet')
        # heatmap = cmap(conv)
        # heatmap = np.delete(heatmap, -1, axis=2)
        # heatmap = (heatmap*256).astype("uint8")
        # heatmap = ((heatmap - np.min(heatmap))/np.ptp(heatmap)*256).astype("uint8")
        back_face = ((faces[0] - np.min(faces[0])) /
                     np.ptp(faces[0]) * 256).astype("uint8")
        the_im = (heatmap / 2 + back_face / 2).astype("uint8")
    else:
        the_im = np.ones((300, 200, 3))
        heatmap = 0

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (the_im, heatmap, locs, preds)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
                default="face_detector",
                help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
                default="mask_detector.model",
                help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
                                "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector modl...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 5.0, (600, 300))


# loop over the frames from the video stream
def new_func(frame, startX, startY, endX, endY, color):
    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)


for i in range(200):
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (im, heat, locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    # show the output frame

    im = cv2.resize(im, (200, 300))
    test = np.ones((300, 200))
    frame = np.concatenate((frame, im), axis=1)

    out.write(frame)
    cv2.imshow("Frame", frame)
    # cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup

# cap.release()
out.release()
cv2.destroyAllWindows()
vs.stop()
