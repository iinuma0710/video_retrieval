from clipping import clip_video
from detection import Darknet

import cv2
import pprint
from PIL import Image


darknet = Darknet(
    config_file="./darknet/cfg/yolov4.cfg",
    weight_file="./darknet/weights/yolov4.weights",
    meta_file="./darknet/cfg/coco.data",
    batch_size=16
)

# video_path = "./jupyter/shot0_1001.mp4"
video_path = "/net/per610a/export/das18a/satoh-lab/share/datasets/kinetics600/video/train/walking_the_dog/0sL5rRoMgLs_000015_000025.mp4"
video_clip_list = clip_video(video_path)

# for i in range(len(video_clip_list[0])):
#     cv2.imwrite("imgs/test{}.jpg".format(str(i)), video_clip_list[0][i])
    # im = Image.fromarray(video_clip_list[0][i])
    # im.save("imgs/test_{}.jpg".format(str(i)))

# for img in video_clip_list[0]:
#     pprint.pprint(darknet.detect([img])) 

pprint.pprint(darknet.detect_batch(video_clip_list[0][:16]))