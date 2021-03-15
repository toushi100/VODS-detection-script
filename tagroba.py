#!/home/ahmed/anaconda3/envs/TensorFlow/bin/python3.8
import tensorflow as tf
import tensorflow_hub as hub
import time,imageio,sys,pickle

# sys.argv[1] is used for taking the video path from the terminal
start = time.time()
video = sys.argv[1]
#passing the video file to ImageIO to be read later in form of frames
video = imageio.get_reader(video)
dictionary = {}
#download and extract the model( faster_rcnn/openimages_v4/inception_resnet_v2 or
# openimages_v4/ssd/mobilenet_v2) in the same folder
module_handle = "/home/ahmed/Desktop/SSD/SSD"
detector = hub.load(module_handle).signatures['default']
#looping over every frame in the video
for index, frames in enumerate(video):
    # converting the images ( video frames ) to tf.float32 which is the only acceptable input format
    image = tf.image.convert_image_dtype(frames, tf.float32)[tf.newaxis]
    # passing the converted image to the model
    detector_output = detector(image)
    class_names = detector_output["detection_class_entities"]
    scores = detector_output["detection_scores"]
    # in case there are multiple objects in the frame
    for i in range(len(scores)):
        if scores[i] > 0.3:
            #converting form bytes to string
            object = class_names[i].numpy().decode("ascii")
            #adding the objects that appear in the frames in a dictionary and their frame numbers
            if object not in dictionary:
                dictionary[object] = [index]
            else:
                dictionary[object].append(index)

with open('filename.pickle', 'wb') as handle:
    pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
end = time.time()
print(end - start)


