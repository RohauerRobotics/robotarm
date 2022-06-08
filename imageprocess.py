# conda activate yolov4-gpu --> activates env for running tf and dependencies

# Run yolov4 deep sort object tracker on video
# $python object_tracker.py --video ./data/video/test.mp4 --output ./outputs/demo.avi --model yolov4

# Run yolov4 deep sort object tracker on webcam (set video flag to 0)
# $python object_tracker.py --video 0 --output ./outputs/webcam.avi --model yolov4

# Run yolov4 tiny
# $python object_tracker.py --weights ./checkpoints/yolov4-tiny-416 --model yolov4 --video ./data/video/test.mp4 --output ./outputs/tiny.avi --tiny

# Run Webcam on Yolov4 tiny
# $python object_tracker.py --weights ./checkpoints/yolov4-tiny-416 --model yolov4 --video 0 --output ./outputs/tiny-cam.avi --tiny

import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
# flags used to define program inputs
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
# import packages for QUEUE multiprocessing
from multiprocessing import Lock, Process, Queue, current_process
import queue

# definitions for flag input variables

flags = {'framework':'tf','weights':'./checkpoints/yolov4-416','size':416, 'tiny':False,'model':'yolov4',
'video':0,'output':None,'output_format':'XVID','iou': 0.45,'score': 0.50,'dont_show':False,'info':False,
'count':False,'count':False}

class Image_Processing(object):
    def __init__(self):
        pass

    def obj_search(self,name):
        dat = [False]
        print("What ",name ," are you looking for?")
        obj = str(input())
        class_names, lst_names = utils.read_class_names(cfg.YOLO.CLASSES)
        # print(lst_names)
        if obj not in lst_names:
            print("name not in class options")
        elif obj in lst_names:
            print("Searching for: ", obj)
            w = self.image_search(obj, 1000)
            if (w[1] != None):
                print("Found:",w[0],"at: ",w[1])
                dat.append(w)
                dat[0] = True
            else:
                print("Nothing was found")
        else:
            pass
        return dat

    def initialization_move_to(self,dat):
        print("""Please enter which of the following you'd like to do:
        1 - Move object to custom (x,y,z) location
        2 - Search for location to move to
        3 - Cancel """)
        sel = str(input("Input: "))
        travel = [False]
        if (sel == "1"):
            x = int(input("Enter x coordinate for end effector in mm: "))
            y = int(input("Enter y coordinate for end effector in mm: "))
            z = int(input("Enter z coordinate for end effector in mm: "))
            dest = [x,y,z]
            print("Moving to :", dest)
            # validate_endeffector()
            travel = [dat,dest]
        elif (sel == "2"):
            dest = self.obj_search("location")
            if dest[0]:
                print("Moving to :", dest[1])
                travel = [dest[0],dat[1],dest[1]]
            else:
                print("Improper destination")
        elif (sel == "3"):
            print("Canceling... ")
        else:
            print("Invalid Option")
        return travel
        # form of the dat list
        # dat[1] = ['remote', [335, 129, 409, 222]]

    def image_search(self,obj, iter):
        # Definition of the parameters
        max_cosine_distance = 0.4
        nn_budget = None
        nms_max_overlap = 1.0

        # initialize deep sort
        model_filename = 'model_data/mars-small128.pb'
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        # calculate cosine distance metric
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        # initialize tracker
        tracker = Tracker(metric)

        # load configuration for object detector
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(flags)
        # input size defined
        input_size = flags['size']
        # video path defined by input flag
        video_path = flags['video']
        # load tflite model if flag is set
        if flags['framework'] == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=flags['weights'])
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(input_details)
            print(output_details)
        # otherwise load standard tensorflow saved model
        else:
            saved_model_loaded = tf.saved_model.load(flags['weights'], tags=[tag_constants.SERVING])
            infer = saved_model_loaded.signatures['serving_default']
        # video_path = 1
        # begin video capture
        try:
            vid = cv2.VideoCapture(int(video_path))
        except:
            vid = cv2.VideoCapture(video_path)

        out = None

        # get video ready to save locally if flag is set
        # if flags['output']:
        #     # by default VideoCapture returns float instead of int
        #     width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        #     height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #     fps = int(vid.get(cv2.CAP_PROP_FPS))
        #     codec = cv2.VideoWriter_fourcc(*flags['output']_format)
        #     out = cv2.VideoWriter(flags['output'], codec, fps, (width, height))

        frame_num = 0
        obj_loc = None
        # while video is running
        while (frame_num < iter):
            if obj_loc != None:
                break
            else:
                pass
            # video input is from
            return_value, frame = vid.read()
            if return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
            else:
                print('Video has ended or failed, try a different video format!')
                break
            frame_num +=1
            # print('Frame #: ', frame_num)
            frame_size = frame.shape[:2]
            # image is processed to be processed
            image_data = cv2.resize(frame, (input_size, input_size))
            # print("INPUT SIZE:", input_size)
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            #
            start_time = time.time()

            # run detections on tflite if flag is set
            if flags['framework'] == 'tflite':
                interpreter.set_tensor(input_details[0]['index'], image_data)
                interpreter.invoke()
                pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                # run detections using yolov3 if flag is set
                if flags['model'] == 'yolov3' and flags['tiny'] == True:
                    boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
                else:
                    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
            else:
                batch_data = tf.constant(image_data)
                pred_bbox = infer(batch_data)
                for key, value in pred_bbox.items():
                    boxes = value[:, :, 0:4]
                    pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=flags['iou'],
                score_threshold=flags['score']
            )

            # convert data to numpy arrays and slice out unused elements
            num_object = valid_detections.numpy()[0]
            bboxes = boxes.numpy()[0]
            bboxes = bboxes[0:int(num_object)]
            scores = scores.numpy()[0]
            scores = scores[0:int(num_object)]
            classes = classes.numpy()[0]
            classes = classes[0:int(num_object)]

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(bboxes, original_h, original_w)

            # store all predictions in one parameter for simplicity when calling functions
            pred_bbox = [bboxes, scores, classes, num_object]

            # read in all class names from config
            class_names,lst_names = utils.read_class_names(cfg.YOLO.CLASSES)

            # by default allow all classes in .names file
            allowed_classes = list(class_names.values())

            # custom allowed classes (uncomment line below to customize tracker for only people)
            # allowed_classes = ['orange']

            # loop through object and use class index to get class name, allow only classes in allowed_classes list
            names = []
            deleted_indx = []
            for i in range(num_object):
                class_indx = int(classes[i])
                class_name = class_names[class_indx]
                if class_name not in allowed_classes:
                    deleted_indx.append(i)
                else:
                    names.append(class_name)
            names = np.array(names)
            count = len(names)
            if flags['count']:
                cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
                print("Objects being tracked: {}".format(count))
            # delete detections that are not in allowed_classes
            bboxes = np.delete(bboxes, deleted_indx, axis=0)
            scores = np.delete(scores, deleted_indx, axis=0)

            # encode yolo detections and feed to tracker
            features = encoder(frame, bboxes)
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

            #initialize color map
            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            # run non-maxima supression
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            # update tracks
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                class_name = track.get_class()

            # draw bbox on screen
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

            # if enable info flag then print details about each track
                if flags['info']:
                    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
                print("Looking for: ", obj)
                print("Obj:",class_name)
                if (class_name == obj):
                    print("Found: ", obj)
                    obj_loc = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                    break
                else:
                    pass

            # calculate frames per second of running detections
            fps = 1.0 / (time.time() - start_time)
            # print("FPS: %.2f" % fps)
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # pass coordinates to other program
            # move_to = frame_num

            if not flags['dont_show']:
                cv2.imshow("Output Video", result)
                # print("Image Dimensions: ", result.shape)

            # if output flag is set, save video file
            if flags['output']:
                out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        session.close()
        cv2.destroyAllWindows()
        return [obj,obj_loc]

    def main(self):
        dat = self.obj_search('object')
        if dat[0]:
            travel = self.initialization_move_to(dat)
            print("Travel Path Created")
            print(np.matrix(travel))
        else:
            pass


# nan = Image_Processing()
# nan.main()
#
# if __name__ == '__main__':
#     try:
#         main()
#     except SystemExit:
#         pass
