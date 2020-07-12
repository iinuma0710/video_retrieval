import numpy as np
import cv2


# 動画ファイルを開く
video = cv2.VideoCapture('/net/per610a/export/das18a/satoh-lab/share/datasets/eastenders/video_detected/shot0_24_id_2.mp4')

# PCに接続されたカメラの映像を表示
# video = cv2.VideoCapture(0)

if not video.isOpened():
    raise RuntimeError

cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)

while True:

    ok, frame = video.read()

    if not ok:
        break

    cv2.imshow('frame', frame)

    key = cv2.waitKey(int(1000 / 30))

    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()