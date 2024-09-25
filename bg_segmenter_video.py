import os
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp
from multiprocessing import Pool
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

class MediapipeSegmenter:
    def __init__(self):
        model_path = 'data_gen/utils/mp_feature_extractors/selfie_multiclass_256x256.tflite'
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            print("Downloading segmenter model from Mediapipe...")
            os.system(f"wget https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite")
            os.system(f"mv selfie_multiclass_256x256.tflite {model_path}")
            print("Download success")

        base_options = BaseOptions(model_asset_path=model_path)
        self.options = vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True
        )

    def segment_image(self, img):
        segmenter = vision.ImageSegmenter.create_from_options(self.options)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        out = segmenter.segment(mp_image)
        return out.category_mask.numpy_view().copy()

    def apply_green_screen(self, img, segmap, segment_head_only = False):
        green_background = np.zeros_like(img)
        green_background[..., 1] = 255  # Set green channel to 255

        if segment_head_only:
            mask = np.isin(segmap, [1, 3, 5])
        else:
            mask = segmap > 0

        green_screen_img = np.where(mask[:, :, None], img, green_background)
        return green_screen_img

def process_frame(frame_info):
    frame, frame_num, seg_model = frame_info
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    segmap = seg_model.segment_image(img)
        # 0 - background
        # 1 - hair
        # 2 - body-skin
        # 3 - face-skin
        # 4 - clothes
        # 5 - others (accessories)

    ### Extra processing for Don Video: Find a mask with all pixels with value 3, find it's upper 2/3rd and correct the body pixels
    # face_mask = segmap == 3
    # y, x = np.where(face_mask)
    # y_min, y_max = y.min(), y.max()
    # x_min, x_max = x.min(), x.max()
    # y_two_third = y_min + 2*(y_max - y_min) // 3
    # body_mask = segmap == 2
    # body_mask[y_two_third:, :] = 0
    # segmap[body_mask] = 3

    green_screen_img = seg_model.apply_green_screen(img, segmap) #, segment_head_only=True)
    return frame_num, cv2.cvtColor(green_screen_img, cv2.COLOR_RGB2BGR)

def process_video(video_path, output_path, seg_model):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Frame width: {frame_width}, Frame height: {frame_height}, FPS: {fps}, Frame count: {frame_count}")

    save_as_video = False
    if output_path.endswith('.mp4'):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
        out_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        save_as_video = True
        print(f"Saving output video at {output_path}")
    else:
        os.makedirs(output_path, exist_ok=True)
        print(f"Saving output frames at {output_path}")

    cpu_count = min(32,os.cpu_count())
    print(f"Using {cpu_count} cores for parallel processing")

    frame_info_list = []

    for frame_num in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frame_info_list.append((frame, frame_num, seg_model))

    cap.release()

    with Pool(cpu_count) as p:
        for frame_num, processed_frame in tqdm(p.imap(process_frame, frame_info_list), total=frame_count):
            if save_as_video:
                out_video.write(processed_frame)
            else:
                cv2.imwrite(os.path.join(output_path, f"{frame_num:08d}.png"), processed_frame)

    if save_as_video:
        out_video.release()

if __name__ == '__main__':
    video_path = "Myself.mp4"
    output_path = "datasets/Myself/gt"
    seg_model = MediapipeSegmenter()

    process_video(video_path, output_path, seg_model)
