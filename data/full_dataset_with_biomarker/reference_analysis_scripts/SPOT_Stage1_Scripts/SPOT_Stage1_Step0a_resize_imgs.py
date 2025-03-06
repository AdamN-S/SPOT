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
    
    import SPOT.Utility_Functions.file_io as fio 
    
    
    
    # caspase_video_folder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/21-SPOT-Drug screen-dye/Caspase_or_cytox_dye/Videos'
    # hoescht_video_folder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/21-SPOT-Drug screen-dye/Hoescht_only/Videos' 
    # mastersavefolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/21-SPOT-Drug screen-dye/Caspase_or_cytox_dye/Videos_Resize'
    
    # caspase_video_folder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/26-SPOT-Drug screen-dye/Caspase or Cytotox Only/Videos'
    # hoescht_video_folder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/26-SPOT-Drug screen-dye/Hoescht Only/Videos' 
    # mastersavefolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/26-SPOT-Drug screen-dye/Caspase or Cytotox Only/Videos_Resize'
    
    # caspase_video_folder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/27-SPOT-Drug screen-dye/Caspase or Cytotox Only/Videos'
    # hoescht_video_folder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/27-SPOT-Drug screen-dye/Hoescht Only/Videos' 
    # mastersavefolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/27-SPOT-Drug screen-dye/Caspase or Cytotox Only/Videos_Resize'
    
    caspase_video_folder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/30-SPOT-Drug screen-dye/Caspase or Cytotox Only/Videos'
    hoescht_video_folder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/30-SPOT-Drug screen-dye/Hoescht Only/Videos' 
    mastersavefolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/30-SPOT-Drug screen-dye/Caspase or Cytotox Only/Videos_Resize'
    
    fio.mkdir(mastersavefolder)
    
    
    """
    Video detection and loading
    """
    
    brightfield_vidfiles = np.sort(glob.glob(os.path.join(caspase_video_folder,
                                                      '*BF.AVI')))
    caspase_vidfiles = np.hstack([ff.replace(' BF.AVI', '.AVI') for ff in brightfield_vidfiles]) # replicated for the dyes..? 
    
    print(brightfield_vidfiles)


    """
    Read the files in. 
    """    
    # these are paired. # we can just show the principle with both marker. 
    vid = fio.read_video_cv2(brightfield_vidfiles[0])
    vid_caspase = fio.read_video_cv2(caspase_vidfiles[0])
    
    print('image shape, ', vid_caspase.shape)
    
    
    """
    Pre-resize the images. (the neural nets are trained at 512x512, therefore it will process faster if downsized.)
    """
    
    # specify the export folder (we will work with this. )
    
    caspase_ch = 0 # the first red channel.
    
    for vid_ii in tqdm(np.arange(len(brightfield_vidfiles))[:]):
        
        basename = os.path.split(brightfield_vidfiles[vid_ii])[-1]
        basename_caspase = os.path.split(caspase_vidfiles[vid_ii])[-1]
        
        vid = fio.read_video_cv2(brightfield_vidfiles[vid_ii])
        
        if len(vid.shape) == 4:
            vid = vid[...,0]

        vid_caspase = fio.read_video_cv2(caspase_vidfiles[vid_ii])
        
        if len(vid_caspase.shape) == 4:
            vid_caspase = vid_caspase[...,caspase_ch]
        
        
        vid_resize = np.uint8(np.array([sktform.resize(vv, output_shape=(512,512), order=1, preserve_range=True) for vv in vid]))
        vid_caspase_resize = np.uint8(np.array([sktform.resize(vv, output_shape=(512,512), order=1, preserve_range=True) for vv in vid_caspase]))
        
    
        skio.imsave(os.path.join(mastersavefolder,
                                 basename+'.tif'), 
                    vid_resize)
        
        skio.imsave(os.path.join(mastersavefolder,
                                 basename_caspase+'.tif'), 
                    vid_caspase_resize)
        
    
    
    
    
    
    
    
    
    