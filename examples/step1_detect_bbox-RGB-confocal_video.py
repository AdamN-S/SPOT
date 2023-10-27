# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 01:51:28 2022

@author: fyz11
"""

if __name__=="__main__":
    
    import numpy as np 
    import SAM.Image.image as SAM_image
    import SAM.Utility_Functions.file_io as fio
    import SAM.Detection.detection as detection 
    
    imfile = r'C:\Users\fyz11\Documents\Work\Projects\Lu-Organoids\Datasets\APC, KRAS, p53 Take 10 25.02.2021 - 384-well (No for KRAS)\p53 Null 4.wmv'
    # imfile = r'C:\Users\fyz11\Documents\Work\Projects\Lu-Organoids\Datasets\APC, KRAS, p53 Take 10 25.02.2021 - 384-well (No for KRAS)\APCmin 1.wmv'

    vid = fio.read_video_cv2(imfile)
    
    
    """
    Specify the trained detection model - YOLOv3
    """
    # vid_unmixed = SAM_image.spectral_unmix_RGB_video(vid, alpha=1, l1_ratio=.5)
    
    outfolder = 'test_detection_folder'
    weightsfile = r'C:\Users\fyz11\Documents\Work\Projects\Lu-Organoids\Scripts\2022-09-07_scripts_for_keras\keras_YOLOv3_organoid_detector2.h5'
    
    detection.load_and_run_YOLOv3_weights_keras_detection_model_RGB(vid, 
                                                                    weightsfile,
                                                                    outfolder,  
                                                                    obj_thresh=0.001, 
                                                                    nms_thresh = 0.45,
                                                                    imsize=(512,512),  
                                                                    anchors=None,
                                                                    class_name_file=None)