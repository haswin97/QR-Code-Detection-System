# import the necessary packages
from tkinter import Label #library utk tampilkan teks/keterangan
from pyzbar import pyzbar #library utk membaca QR Code
import datetime #library utk menampilkan tanggal & jam

import time
import cv2
import os #library utk buka file
import numpy as np #library utk mengubah gambar ke array
import tensorflow.compat.v1 as tf #konversi TF version 2 ke TF version 1
import sys #library utk

from PIL import ImageTk as imtk #Python Imaging Library utk image processing di GUI
from PIL import Image as img

import globalvar

# Import utilites
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


def resize_image(image, const_width):
    height, width, channels = image.shape

    scale = float(const_width) / float(height)
    re_width = int(width * scale)
    re_height = const_width
    dim = (re_width, re_height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    return resized


def show_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #utk convert BGR ke RGB

    img_resize = resize_image(image_rgb, globalvar.CONST_WIDTH)
    img_frame = img.fromarray(img_resize)
    img_show = imtk.PhotoImage(img_frame)

    try:
        globalvar.label_video.destroy()
    except:
        print("Continue Playing a Video ...")

    globalvar.label_video = Label(globalvar.root, image=img_show)
    globalvar.label_video.place(x=28, y=50)

    globalvar.root.update()


def process_detection(video_path):
    detected = object_detection(video_path)
    return detected


def object_detection(video_path):
    info_status = "Status : Configure Tensorflow ... Please Wait"
    globalvar.label_status.configure(text=info_status, font='Arial 9 bold')
    globalvar.root.update()

    globalvar.result_qrcode = ""
    tf.disable_v2_behavior

    detected = False
    time.sleep(1.0) #utk beri jarak CPU proses tensorflow

    # Grab path to current working directory
    CWD_PATH = os.getcwd()
    OBJECT_DETECTION_PATH = os.path.join(CWD_PATH, 'object_detection')

    output_path = os.path.join(CWD_PATH,"barcodes.csv")
    # open the output CSV file for writing and initialize the set of
    # qrcodes found thus far
    csv = open(output_path, "w")
    found = set()

    # Name of the directory containing the object detection module we're using
    MODEL_NAME = 'inference_graph_16K'

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(OBJECT_DETECTION_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

    # Path to label map file
    PATH_TO_LABELS = os.path.join(OBJECT_DETECTION_PATH, 'annotations', 'labelmap.pbtxt')

    # Path to video
    PATH_TO_VIDEO = video_path

    # Number of classes the object detector can identify
    NUM_CLASSES = 1

    try:
        # Load the label map.
        # Label maps map indices to category names, so that when our convolution
        # Here we use internal utility functions, but anything that returns a
        # dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                        use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        print('Load Label Map Successfully...')
    except:
        print('Load Label Map Error...')
        return False

    try:
        # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            sess = tf.Session(graph=detection_graph)
        print('Load The Tensorflow Model Successfully ...')
    except:
        print('Load The Tensorflow Model Error ...')
        return False

    try:
        # Define input and output tensors (i.e. data) for the object detection classifier

        # Input tensor is the image
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

        # Number of objects detected
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        print('Define Input and Output Tensors ...')
    except:
        print('Input and Output Tensors Error ...')
        return False


    print("Playing Video ...")
    info_status = "Status : Playing Video ..."
    globalvar.label_status.configure(text=info_status, font='Arial 9 bold')
    globalvar.root.update()

    # Open video file
    cap = cv2.VideoCapture(video_path)
    time.sleep(1.0) #utk delay video

    # loop over the frames from the video
    while cap.isOpened():
        stime = time.time() #utk FPS
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value

        ret, frame = cap.read()

        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            qrcodes = pyzbar.decode(thresh)

            # loop over the detected qrcodes
            for qrcode in qrcodes:

                # extract the bounding box location of the qrcode and draw
                # the bounding box surrounding the qrcode on the video
                (x, y, w, h) = qrcode.rect
                #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

                # the qrcode data is a bytes object so if we want to draw it
                # on our output image we need to convert it to a string first
                qrcodeData = qrcode.data.decode("utf-8")
                qrcodeType = qrcode.type

                # draw the qrcode data and qrcode type on the video
                text = "{}".format(qrcodeType)
                cv2.putText(frame, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                info_status = "Status : QR-Code Detected ..."
                now = datetime.datetime.now()

                info_detected = "Detected on: {}".format(now.strftime("%d/%m/%Y, %H:%M:%S"))
                print(info_detected)
                globalvar.result_qrcode = "{}\n\n{}\n\n----------------------------------\n\n".format(
                    info_detected, qrcodeData)
                detected = True

                # if the qrcode text is currently not in our CSV file, write
                # the timestamp + barcode to disk and update the set
                if qrcodeData not in found:
                    csv.write(globalvar.result_qrcode)
                    csv.flush()
                    found.add(qrcodeData)
            #
            # frame_expanded = np.expand_dims(frame, axis=0)
            #
            # # Perform the actual detection by running the model with the image as input
            # (boxes, scores, classes, num) = sess.run(
            #     [detection_boxes, detection_scores, detection_classes, num_detections],
            #     feed_dict={image_tensor: frame_expanded})
            #
            # # Draw the results of the detection (aka 'visulaize the results')
            # vis_util.visualize_boxes_and_labels_on_image_array(
            #     frame,
            #     np.squeeze(boxes),
            #     np.squeeze(classes).astype(np.int32),
            #     np.squeeze(scores),
            #     category_index,
            #     use_normalized_coordinates=True,
            #     line_thickness=8,
            #     min_score_thresh=0.99)

            # All the results have been drawn on the frame, so it's time to display it.
            # cv2.imshow('Object detector', frame)

            fps_info = 'FPS = {:.1f}'.format(1 / (time.time() - stime))
            globalvar.label_status.configure(text="{} ({})".format(info_status, fps_info), font='Arial 9 bold')
            show_image(frame)
            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            break

    globalvar.label_status.configure(text="Status : Video Ended ...", font='Arial 9 bold')
    return detected


