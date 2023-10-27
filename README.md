# SPOT (Shape, appearance, motion Phenotype Observation Tool)

<p align="center">
  <img src="https://github.com/fyz11/SPOT/blob/main/docs/pictures/main_workflow.jpg" width=100%/>
</p>

## Introduction
SPOT is a generalized and streamlined workflow for analysing object dynamics in movies. It is designed to suit high-content imaging applications where analytical tools should be push-and-go and require no prior knowledge of the expected behaviour of the objects to be studied. In other words, users should be able to run through all the steps in one go, then retrospectively interpret the produced results. This workflow is inspired by that for single-cell sequencing and is summarized in the figure above.  


SPOT is provided here as a Python package to allow full flexibility. To get started, please check out exemplar scripts in the examples/ folder. We also include SPOTapp, a graphical user interface (GUI) to run SPOT stages 1 and 2. This app is provided as is in alpha development. It will be separately developed and maintained at the following address: XXXX.   

## Associated Paper
SPOT is associated with the following paper, which you can read for more technical detail and get an idea of the applications SPOT enables: 

**Measuring and monitoring complex dynamic phenotypic heterogeneity in live-cell imaging using Shape, Appearance and Motion Phenotype Observation Tool (SPOT)**, 2023, written by Felix Y. Zhou, Brittany-Amber Jacobs, Xiaoyue Han, Adam Norton-Steele, Thomas M. Carroll, Carlos Ruiz Puig, Joseph Chadwick, Xiao Qin, Richard Lisle, Lewis Marsh, Helen M. Byrne, Heather A. Harrington, Linna Zhou and Xin Lu.

## Getting Started
Exemplar scripts to run every step of the workflow are provided in the Examples folder which uses the example data located in the data/ folder. This includes:

**for SPOT Stage 1: video acquisition and object segmentation:**
1. SPOT_Stage1_Step0_unmix-RGB-confocal_video.py
2. SPOT_Stage1_Step1_detect_bbox-RGB-confocal_video.py
3. SPOT_Stage1_Step2_track_detect_bbox-RGB-confocal_video.py
4. SPOT_Stage1_Step3_segment_tracked_bbox-RGB-confocal_video.py
5. SPOT_Stage1_Step4_postprocess_segment_tracked_bbox-RGB-confocal_video.py

**for SPOT Stage 2: computation of SAM (Shape, Appearance and Motion) phenome:**

**for SPOT Stage 3: analysis of SAM (Shape, Appearance and Motion) phenome:**
1. SPOT_Stage3_Example-cell_tracking_challenge_dataset.py


## Pretrained Neural Network Models for organoid detection and segmentation


## Documentation
Documented API of the functions provided in this library is available as a html in the docs/build/hmtl folder. You can build up-to-date docs by going into docs/ and executing:
```shell
make html
```

## To install
The majority of the requirements of the package can be installed using pip:
```shell
pip install .
```
You will need to install the demeter package from source in order to compute Euler Characteristic Curve (ECC) shape features:

* download demeter with:
```shell
git clone https://github.com/amezqui3/demeter/
```
* move to the directory:
```shell
cd demeter
```
* install with:
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
