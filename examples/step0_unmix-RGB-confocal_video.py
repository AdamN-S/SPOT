# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 01:51:28 2022

@author: fyz11
"""

if __name__=="__main__":
    
    import numpy as np 
    import SAM.Image.image as SAM_image
    import SAM.Utility_Functions.file_io as fio
    
    imfile = r'C:\Users\fyz11\Documents\Work\Projects\Lu-Organoids\Datasets\APC, KRAS, p53 Take 10 25.02.2021 - 384-well (No for KRAS)\p53 Null 4.wmv'
    
    vid = fio.read_video_cv2(imfile)
    
    
    """
    spectral unmix 
    """
    vid_unmixed = SAM_image.spectral_unmix_RGB_video(vid, alpha=1, l1_ratio=.5)