#!/home/ahmed/anaconda3/envs/tensorflow_cpu/bin/python3.6
import imageio
import numpy as np
import os
import tarfile
import tensorflow as tf
from datetime import datetime
from object_detection.utils import label_map_util
import json
import sys

from object_detection.utils import visualization_utils as vis_util

PATH_FROM_WEB = '/home/ahmed/Desktop/cocov1/'
model_name = PATH_FROM_WEB+'ssd_mobilenet_v1_coco_2018_01_28'
model_file = model_name + '.tar.gz'
PATH_TO_CKPT = model_name + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join(
    PATH_FROM_WEB+'object_detection/data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90
min_score_thresh = 0.5
tarfile = tarfile.open(model_file)
for file in tarfile.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tarfile.extract(file, os.getcwd())

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


with detection_graph.as_default():
    with tf.Session(graph=detection_graph)as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name(
            'detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name(
            'detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name(
            'detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        videoname = sys.argv[1]
        input_video = '/home/ahmed/Desktop/blog/storage/app/public/videos/'+videoname
        video_reader = imageio.get_reader('%s.mp4' % input_video)
        video_writer = imageio.get_writer(
            '%s_annotated.mp4' % input_video, fps=30)

        t0 = datetime.now()
        n_frames = 0
        case_list = []
        for frame in video_reader:
            image_np = frame
            n_frames += 1
            image_np_expanded = np.expand_dims(image_np, axis=0)
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores,
                    detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            video_writer.append_data(image_np)
            final_score = np.squeeze(scores)
            count = 0
            for i in range(100):
                if scores is None or final_score[i] > 0.5:
                    count = count + 1

            for x in range(count):
                dictionary = {'object': category_index[classes[0][x]]['name'], 'frame': n_frames}
                case_list.append(dictionary)

        fps = n_frames / (datetime.now() - t0).total_seconds()
        video_writer.close()

        #print(case_list)
        json_object = json.dumps(case_list, indent=2)
        print(json_object)

        # Writing to sample.json
        with open("the json file.json", "w") as outfile:
            outfile.write(str(json_object))
