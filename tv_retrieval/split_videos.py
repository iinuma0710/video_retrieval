import os
import cv2
import glob
import numpy as np

THRESH = 55.55679398148148
TV_STATIONS = ["bs1", "etv", "fuji", "net", "NHK", "ntv", "tbs", "tvtokyo"]

def MAE(pic):
    return np.mean(np.abs(pic))


# 映像の書き出しを行う
def write_video(shot_path, frames, fps, w, h):
    if frames == []:
        return

    fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
    mv = cv2.VideoWriter(shot_path, fourcc, fps, (w, h))
    for f in frames:
        mv.write(f)
    mv.release()


def split_video(video, output_dir):
    cap = cv2.VideoCapture(video)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    file_name = os.path.basename(video)[:-4]

    frame_cnt = 0
    # フレーム画像を比較用
    pic_size = (64, 36)
    prev_frame = np.zeros((*pic_size[::-1], 3))
    # 書き出すフレームを格納
    shot_num = 0
    frames = []
    while True:
        ret, frame = cap.read()
        if ret:
            # フレーム画像を縮小して，前のフレームとのMAEを計測
            curr_frame = cv2.resize(frame, pic_size, interpolation=cv2.INTER_AREA)
            mae = MAE(curr_frame.astype(np.int) - prev_frame.astype(np.int))

            # 前のフレームとの差分が一定値以上なら映像を書き出す
            if mae >= THRESH:
                print("Cut detected!: frame {}".format(frame_cnt))
                shot_path = os.path.join(output_dir, file_name + "_shot_{}.mp4".format(shot_num))
                write_video(shot_path, frames, fps, w, h)
                # 書き出すフレームの情報を設定
                shot_num += 1
                frames = [frame]
            else:
                frames.append(frame)

            # 次にフレームに進む
            frame_cnt += 1
            prev_frame = curr_frame
        else:
            break

    
if __name__ == "__main__":
    for station in TV_STATIONS:
        # 入力と出力のディレクトリを指定
        input_dir = "./data/recorded/{}/2021/2021_02_07_04_56/".format(station)
        output_dir = "./data/shots/{}/2021/2021_02_07_04_56/".format(station)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 入力のディレクトリに格納されている映像ファイルの一覧を取得
        station_videos = glob.glob(os.path.join(input_dir, "*.mp4"))

        # 映像の分割
        for video in station_videos:
            print("Now processing : ", video)
            split_video(video, output_dir)

    # output_dir = "./data/shots/NHK/2021/2021_02_07_04_56/"
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # video = "./data/recorded/NHK/2021/2021_02_07_04_56/2021_02_07_04_56_00.mp4"
    # split_video(video, output_dir)