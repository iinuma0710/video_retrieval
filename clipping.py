import cv2
import math
import numpy as np


def clip_video(video_path):
    # 映像のプロパティを取得
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 1ビデオクリップのフレーム数を決める
    if frame_count >= 64:
        frames_per_clip = 64
    elif frame_count < 64 and frame_count >= 32:
        frames_per_clip = 32
    else:
        print("This viodeo is too short...")
        return None
    clip_num = math.ceil(frame_count / frames_per_clip)
    duplicate_frams = [0] + [(frames_per_clip * clip_num - frame_count + i) // (clip_num - 1) for i in range(clip_num - 1)]
    
    # フレーム画像の取得
    frame_image_list = []
    while True:
        ret, frame = cap.read()
        if ret:
            frame_image_list.append(frame)
        else:
            break
    
    # ビデオクリップのフレームの番号を選択
    frame_idx_list = []
    for clip_idx in range(clip_num):
        first_frame_idx = clip_idx * frames_per_clip - sum(duplicate_frams[:clip_idx + 1])
        if frames_per_clip == 64:
            frame_idx = [first_frame_idx + i * 2 for i in range(32)]
        else:
            frame_idx = [first_frame_idx + i for i in range(32)]
        frame_idx_list.append(frame_idx)
        
    # ビデオクリップを作成する
    video_clip_list = []
    for frame_indexes in frame_idx_list:
        frame_images = [cv2.cvtColor(frame_image_list[idx], cv2.COLOR_BGR2RGB) for idx in frame_indexes]
        video_clip_list.append(frame_images)
        
    return video_clip_list


if __name__ == "__main__":
    video_clip_list = clip_video("./data/test.mp4")
    print("ビデオクリップ数 : ", len(video_clip_list))
    print("ビデオクリップあたりのフレーム数 : ", len(video_clip_list[0]))
    print("映像のテンソル形状 : ", video_clip_list[0][0].shape)