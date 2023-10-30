# SPOT (Shape, appearance, motion Phenotype Observation Tool)

<p align="center">
  <img src="https://github.com/fyz11/SPOT/blob/main/docs/pictures/main_workflow.jpg" width=100%/>
</p>

## Introduction to SPOT
SPOT is a generalized and streamlined workflow for analysing object dynamics in movies. It is designed to suit high-content imaging applications where analytical tools should be push-and-go and require no prior knowledge of the expected behaviour of the objects to be studied. In other words, users should be able to run through all the steps in one go, then retrospectively interpret the produced results. This workflow is inspired by that for single-cell sequencing and is summarized in the figure above. 

Three innovations drive SPOT for temporal image analysis:
1. a standardized Shape, Appearance and Motion phenome - a single feature set for all objects
2. a standardized temporal analysis of compiled SAM phenomes - minimal assumption, push-and-go
3. automated and standardized techniques to cluster related SAM features into SAM modules for interpreting discovered phenotypes

SPOT is provided here as a Python package to allow full flexibility. To get started, please check out exemplar scripts in the examples/ folder. We also include SPOTapp, a graphical user interface (GUI) to run SPOT stages 1 and 2. This app is currently in alpha development and is provided as-is. It will be separately developed and maintained at the following address: XXXX.   

<p align="center">
  <img src="https://github.com/fyz11/SPOT/blob/main/docs/pictures/SAM_motivation.jpg" width=100%/>
</p>

## Introduction to the SAM phenome
Dynamic objects constantly change their behaviour. Motivated by observation of natural images like birds and cars, we hypothesize that three measurable properties; Shape, Appearance and Motion (SAM) provides complementary information necessary to characterize the instantaneous phenotypic state for any object.  

This led us to design a single generalized SAM feature set which can function similar to the single-cell transcriptome in single-cell sequencing analysis by considering in addition to shape, appearance and motion; global, (local) regional and (local) distribution features.


## Associated Paper
SPOT is associated with the following paper, which you can read for more technical detail and get an idea of the many applications SPOT enables: 

**Measuring and monitoring complex dynamic phenotypic heterogeneity in live-cell imaging using Shape, Appearance and Motion Phenotype Observation Tool (SPOT)**, 2023, written by Felix Y. Zhou, Brittany-Amber Jacobs, Xiaoyue Han, Adam Norton-Steele, Thomas M. Carroll, Carlos Ruiz Puig, Joseph Chadwick, Xiao Qin, Richard Lisle, Lewis Marsh, Helen M. Byrne, Heather A. Harrington, Linna Zhou and Xin Lu.

## Getting Started
Exemplar scripts to run every step of the workflow are provided in the Examples folder which uses the example data located in the data/ folder. This includes:

**for SPOT Stage 1: video acquisition and object segmentation:**
1. SPOT_Stage1_Step0_translation-register-RGB-confocal_video.py
2. SPOT_Stage1_Step0_unmix-RGB-confocal_video.py
3. SPOT_Stage1_Step1_detect_bbox-RGB-confocal_video.py
4. SPOT_Stage1_Step2_track_detect_bbox-RGB-confocal_video.py
5. SPOT_Stage1_Step3_segment_tracked_bbox-RGB-confocal_video.py
6. SPOT_Stage1_Step4_postprocess_segment_tracked_bbox-RGB-confocal_video.py

**for SPOT Stage 2: computation of SAM (Shape, Appearance and Motion) phenome:**
1. SPOT_Stage2_Step1_compute_SAM_phenomes.py
2. SPOT_Stage2_Step2_generate_metadata_table.py
3. SPOT_Stage2_Step3_compile_and_export_SAM_phenomes.py

**for SPOT Stage 3: analysis of SAM (Shape, Appearance and Motion) phenome:**
1. SPOT_Stage3_Example-cell_tracking_challenge_dataset.py

## Pretrained Neural Network Models for organoid detection and segmentation
We make available pretrained neural network organoid detection and segmentation models with this repo. 

1. Organoid attention UNet segmentation model given a bounding box cropped image (this repo, models/segment_CNN_model/organoid-bbox_attn_seg-unet-master-v2_64x64-8feats_v2.h5' 
2. [Organoid YOLOv3 bounding box detection model weights](https://www.dropbox.com/scl/fi/qzowc9s9n30zh6qdyzeqw/keras_YOLOv3_organoid_detector2.h5?rlkey=6deiqemsmcz3yin9b5dnz0e6y&dl=0)

## Documentation
Documented API of the functions provided in this library is available as a html in the docs/build/hmtl folder. You can build up-to-date docs by going into docs/ and executing:
```shell
make html
```

## To install
The package can be installed using pip:
```shell
pip install .
```

### COPYRIGHT INFORMATION:

#### FOR ACADEMIC AND NON-PROFIT USERS
---
The software and scripts is made available as is under a Ludwig Software License for academic non-commercial research purposes. Please read the included software license agreement.

#### FOR-PROFIT USERS
---
If you plan to use SPOT in any for-profit application, you are required to obtain a separate license. To do so, please contact Shayda Hemmati (shemmati@licr.org) or Par Olsson (polsson@licr.org) at the Ludwig Institute for  Cancer Research Ltd.
