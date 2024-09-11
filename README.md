# ESR

## Setup On Lambda Instance
```bash
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install -y zip unzip curl wget git libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg
cd /home/ubuntu/ESR/Real-ESRGAN/
pip install -r requirements.txt
export PYTHONPATH=./
```

## Training (For 1024 Output)
### Data Preparation
* Put 1024 video frames in `datasets/${VIDEO_ID}/gt/` folder
* Prepare a low-resolution video using `generate_multiscale_data.py` script
* Prepare a meta info file `datasets/${VIDEO_ID}/meta_info_file.txt` using `scripts/generate_meta_info_withface_pairdata.py` script (if you want to use VGG-Face perceptual loss)/`scripts/generate_meta_info_pairdata.py` script (if you don't want to use VGG-Face perceptual loss)
* Create a copy of the configuration file `options/Best_Anime_x4.yml`/`options/Best_Anime_x2.yml` and modify the `name`, `dataroot_gt`, `dataroot_lq`, and `meta_info` accordingly

### Training Commands
**4x Model**
```bash
python realesrgan/train.py -opt options/Best_Anime_x4.yml
```

**2x Model**
```bash
python realesrgan/train.py -opt options/Best_Anime_x2.yml
```

## Inference
### Single Image
```bash
python inference_realesrgan.py --blur_input -n RealESRGAN_x2plus_small --model_path /path/to/experiments/models/net_g.pth -i test_imgs_512/1.png -o Results_x2
```

### Folder
```bash
python inference_realesrgan.py --blur_input -n RealESRGAN_x2plus_small --model_path /path/to/experiments/models/net_g.pth -i test_imgs_512 -o Results_x2
```

### Video
```bash
python3 inference_realesrgan_video.py --blur_input -n RealESRGAN_x2plus_small --model_path /path/to/experiments/models/net_g.pth -i test_video_512.mp4 --output Results_x2_video
```

## Citation
Real-ESRGAN: [GitHub](https://github.com/xinntao/Real-ESRGAN/)
```
@InProceedings{wang2021realesrgan,
    author    = {Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
    title     = {Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
    booktitle = {International Conference on Computer Vision Workshops (ICCVW)},
    date      = {2021}
}
```