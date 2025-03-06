#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 13:51:44 2024

@author: s205272

Analysis of the drugs + dye dataset, with hoescht + caspase staining 

"""


if __name__=="__main__":
    
    import os 
    import glob
    import numpy as np 
    import pylab as plt 
    import skimage.io as skio
    import skimage.transform as sktform 
    from tqdm import tqdm 
    import scipy.io as spio
    
    import SPOT.Utility_Functions.file_io as fio 
    import SPOT.Detection.detection as SPOT_detection
    
    
    """
    We use only the registered brightfield images to detect organoids
    """

    # specify the export folder (we will work with this. )
    # masterimgfolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/21-SPOT-Drug screen-dye/Caspase_or_cytox_dye/Videos_Resize_Register'
    # masterimgfolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/21-SPOT-Drug screen-dye/Caspase_or_cytox_dye_redo/Videos_Resize_Register'
    
    # masterimgfolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/26-SPOT-Drug screen-dye/Caspase or Cytotox Only/Videos_Resize_Register'
    # masterimgfolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/27-SPOT-Drug screen-dye/Caspase or Cytotox Only/Videos_Resize_Register'
    
    masterimgfolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/30-SPOT-Drug screen-dye/Caspase or Cytotox Only/Videos_Resize_Register'

    imgfolders = os.listdir(masterimgfolder)
    imgfolders = np.hstack([os.path.join(masterimgfolder,
                                         ff) for ff in imgfolders if os.path.isdir(os.path.join(masterimgfolder,
                                                                              ff))])


    """
    Specify the CNN detector weights location here. Our pretrained is available and can be downloaded from dropbox. Follow the link in the README.md to download then copy to models/detect_CNN_model/
    """
    # you should download this weights file and place it here. 
    weightsfile = r'/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/Analysis_Scripts_w_Library/Models/detect_CNN_model/keras_YOLOv3_organoid_detector2.h5'

    
    """
    Define settings of the detector 
    """
    obj_thresh = 0.001 # the threshold for calling an organoid 
    nms_thresh = 0.4 # the non max suppression threshold 
    imsize=(512,512) # the image size the detector was trained at which is (512,512)
    equalize_hist = True # whether to perform global histogram equalization, we generally find this improves detection for brightfield and phase-contrast. 
    

    for vid_ii in tqdm(np.arange(len(imgfolders))[:]):
        
        # get the basename. 
        basename = os.path.split(imgfolders[vid_ii])[-1]
        
        # specify the savefolder for this detection part. 
        savefolder_vid_ii = os.path.join(masterimgfolder, 
                                         basename,
                                         'bbox_detection'); 
        fio.mkdir(savefolder_vid_ii)
        
        
        imfile = os.path.join(imgfolders[vid_ii],
                                  basename+'_register.tif')
        
        vid = skio.imread(imfile)
        
        if len(vid.shape) == 4:
            vid = vid[...,0]

        
        # out vid is grayscale, we add an extra channel to fake an RGB 
        vid_RGB = vid[...,None] # 
        
        # =============================================================================
        #   Run detector.       
        # =============================================================================

        """
        Detect bounding boxes using YOLOv3 model and write straight to the output folder. 
        """
        SPOT_detection.load_and_run_YOLOv3_weights_keras_detection_model_RGB(vid_RGB, 
                                                                        weightsfile,
                                                                        savefolder_vid_ii,  
                                                                        obj_thresh=obj_thresh, 
                                                                        nms_thresh = nms_thresh,
                                                                        imsize=imsize,  
                                                                        equalize_hist=equalize_hist,
                                                                        anchors=None,
                                                                        class_name_file=None,
                                                                        debug_viz=True)
        
    
    
    
    
    
    
    