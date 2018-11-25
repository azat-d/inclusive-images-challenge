# inclusive-images-challenge
4th place solution for the [Inclusive Images Challenge on Kaggle](https://www.kaggle.com/c/inclusive-images-challenge).  
Private LB score: **0.33184**
## The hardware I used
- CPU: Intel Core i7 5930k
- GPU: 1x NVIDIA GTX 1080
- RAM: 64 GB
- SSD: 2x 512GB
- HDD: 1x 3TB
## Prerequisites
### Environment
The model was trained in the docker container. It is highly recommended to use [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker) if you want to reproduce the result. 
The code assumes that you have at least 1 NVIDIA GPU and CUDA 8 compatible driver. Run the following command to build the docker image:
```bash
cd path/to/solution
sudo docker build -t inclusive .
```
### Free space
- HDD: ~600 GB (525 GB for the Open Images Training dataset + 71 GB for checkpoints, logs, etc)
- SSD: ~100 GB (77 GB for the resized Open Images Training dataset + 13 GB for the competition data)
### Data
Download the [open-images-dataset](https://www.kaggle.com/c/inclusive-images-challenge#Data-Download-&-Getting-Started) to `/path/to/hdd/open-images-dataset/train`
```bash
mkdir -p /path/to/hdd/open-images-dataset
cd /path/to/hdd/open-images-dataset
aws s3 --no-sign-request sync s3://open-images-dataset/train train/
```
Download the [inclusive-images-challenge-data](https://www.kaggle.com/c/inclusive-images-challenge/data) to `/path/to/ssd/inclusive-images-challenge/data`
```bash
mkdir -p /path/to/ssd/inclusive-images-challenge/data
cd /path/to/ssd/inclusive-images-challenge/data
kaggle competitions download -c inclusive-images-challenge
unzip train_human_labels.csv.zip
unzip stage_1_sample_submission.csv.zip
unzip stage_2_sample_submission.csv.zip
unzip stage_1_test_images.zip -d stage_1_test_images
unzip stage_2_images.zip -d stage_2_test_images
```
_Note: there are some missing files in the Inclusive Images Challenge Stage 1 data. You have to delete them manually from the `tuning_labels.csv`_
### Output directories
Create a directory for the resized Open Images Training dataset
```bash
mkdir -p /path/to/ssd/open-images-dataset/train-resized
```
Create a directory for training artifacts (checkpoints, logs, etc)
```bash
mkdir -p /path/to/hdd/inclusive-images-challenge/artifacts
```
Create a directory for the model output (submissions)
```bash
mkdir -p /path/to/hdd/inclusive-images-challenge/output
```
## How to train the model
Run the docker container with the paths correctly mounted
```bash
sudo docker run --runtime=nvidia -i -t -d --rm --ipc=host -v /path/to/hdd/open-images-dataset:/original_images -v /path/to/ssd/open-images-dataset/train-resized:/resized_images -v /path/to/ssd/inclusive-images-challenge/data:/inclusive -v /path/to/hdd/inclusive-images-challenge/artifacts:/artifacts -v /path/to/hdd/inclusive-images-challenge/output:/output --name inclusive inclusive
sudo docker exec -it inclusive /bin/bash
``` 
Prepare the training dataset (inside the container)
```bash
python3 /source/prepare_dataset.py
```
Train the model on the Open Images Training dataset (inside the container)
```bash
python3 /source/train_se_resnet101.py
```
Finetune the model on the Inclusive Images Challenge Stage 1 tuning set (inside the container)
```bash
python3 /source/tune_se_resnet101.py
```
## How to generate submissions
Run the following command (inside the container)
```bash
python3 /source/generate_submission.py stage_id (1 or 2)
```
Submissions will appear in the output directory: `/path/to/hdd/inclusive-images-challenge/output`  
The fastest way to get predictions for a new test dataset is to replace dataset from the second stage with new one.
## `source/config.yaml`
This file specifies the path to the train, test, model, and output directories.
- This is the only place that specifies the path to these directories.
- Any code that is doing I/O uses the appropriate base paths from `config.yaml`  
_Note: If you are using the docker container, then you do not need to change the paths in this file._
##  Serialized copy of the trained model
You can download my artifacts folder which I used to generate my final submissions: [GoogleDrive](https://drive.google.com/file/d/1rg5m7xKXGdc3jnaI-QKLKtpwUPAmieeP/view?usp=sharing)
