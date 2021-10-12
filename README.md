# extended_version
The resource is from [Pose Classification Options](https://developers.google.com/ml-kit/vision/pose-detection/classifying-poses) about prepossing to get csv file.

Input pose video to images. And using classify images get pose landmarks which put it on record with csv file.

# File description  

**extract_images.py**- Input pose video to images. When extract images done, please attention images(pose class) folder position which is the next one file(csv create.py) to be use.   

**csv_create.py**- using classify images get pose landmarks which put it on record with csv file.  



## Install  

**Conda virtual env**  
```bash

conda create --name [env_name]  python=3.8
conda activate [env_name]
pip install numpy==1.19.3
pip install opencv-python==4.5.1.48
pip install tqdm==4.56.0
pip install mediapipe==0.8.3
pip install pillow==8.1.0
pip install matplotlib==3.3.4
pip install requests==2.25.1
```
