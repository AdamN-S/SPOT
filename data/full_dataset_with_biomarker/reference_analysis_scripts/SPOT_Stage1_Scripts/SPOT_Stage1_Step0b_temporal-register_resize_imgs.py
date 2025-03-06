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
    import SPOT.Registration.registration_2D as SPOT_register
    
    
    
    # videofolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/21-SPOT-Drug screen-dye/Caspase_or_cytox_dye/Videos_Resize'
    # videofolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/21-SPOT-Drug screen-dye/Caspase_or_cytox_dye_redo/Videos_Resize'
    
    # videofolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/26-SPOT-Drug screen-dye/Caspase or Cytotox Only/Videos_Resize'
    # videofolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/27-SPOT-Drug screen-dye/Caspase or Cytotox Only/Videos_Resize'
    
    videofolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/30-SPOT-Drug screen-dye/Caspase or Cytotox Only/Videos_Resize'
    
    """
    Video detection and loading
    """
    
    brightfield_vidfiles = np.sort(glob.glob(os.path.join(videofolder,
                                                      '*BF.AVI.tif')))
    caspase_vidfiles = np.hstack([ff.replace(' BF.AVI.tif', '.AVI.tif') for ff in brightfield_vidfiles]) # replicated for the dyes..? 
    
    print(brightfield_vidfiles)


    """
    Specify the final folders
    """
    # specify the export folder (we will work with this. )
    # mastersavefolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/21-SPOT-Drug screen-dye/Caspase_or_cytox_dye/Videos_Resize_Register'
    # mastersavefolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/21-SPOT-Drug screen-dye/Caspase_or_cytox_dye_redo/Videos_Resize_Register'
    
    # mastersavefolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/26-SPOT-Drug screen-dye/Caspase or Cytotox Only/Videos_Resize_Register'
    # mastersavefolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/27-SPOT-Drug screen-dye/Caspase or Cytotox Only/Videos_Resize_Register'
    
    mastersavefolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/30-SPOT-Drug screen-dye/Caspase or Cytotox Only/Videos_Resize_Register'
    
    fio.mkdir(mastersavefolder)
    

    caspase_ch = 0
    ext = '.tif'
    

    for vid_ii in tqdm(np.arange(len(brightfield_vidfiles))[:]):
        
        basename = os.path.split(brightfield_vidfiles[vid_ii])[-1].split(ext)[0]
        basename_caspase = os.path.split(caspase_vidfiles[vid_ii])[-1].split(ext)[0]
        
        
        savefolder_vid_ii = os.path.join(mastersavefolder, basename); fio.mkdir(savefolder_vid_ii)
        
        vid = skio.imread(brightfield_vidfiles[vid_ii])
        
        if len(vid.shape) == 4:
            vid = vid[...,0]

        vid_caspase = skio.imread(caspase_vidfiles[vid_ii])
        
        if len(vid_caspase.shape) == 4:
            vid_caspase = vid_caspase[...,caspase_ch]
        
        
        # =============================================================================
        #   Perform the registration. 
        # =============================================================================
        
        vid_register, blurred_frames, disps = SPOT_register.translation_register_blurry_phase_contrast_videos(np.uint8(255.*vid/float(vid.max())), 
                                                                                                                              shape=(512,512), 
                                                                                                                              sub_pix_res = 1, 
                                                                                                                              impute_pixels = True, 
                                                                                                                              use_registered = True, 
                                                                                                                              use_dog=False, 
                                                                                                                              dog_sigma=3, 
                                                                                                                              detect_blurred_frames = False, 
                                                                                                                              blur_factor=3., 
                                                                                                                              apply_crop_mask=False)
                                                
        vid_caspase_register = SPOT_register.apply_translation_register_correlation(vid_caspase, 
                                                                                    disps, 
                                                                                    use_registered=True)
        
        """
        visualize the registered movie
        """
        for frame_no in np.arange(len(vid)):
            plt.figure(figsize=(10,10))
            plt.subplot(121)
            plt.title('Frame: ' + str(frame_no).zfill(5) + '_registered')
            plt.imshow(vid_register[frame_no], cmap='gray')
            plt.subplot(122)
            plt.imshow(vid_caspase_register[frame_no], cmap='gray')
            plt.show()
        
        
        # save into individual folder with individal registration parameters. 
        
        skio.imsave(os.path.join(savefolder_vid_ii,
                                 basename+'_register.tif'), 
                    vid_register)
        
        skio.imsave(os.path.join(savefolder_vid_ii,
                                 basename_caspase+'_register.tif'), 
                    vid_caspase_register)
        
        """
        save the registration parameters
        """
        spio.savemat(os.path.join(savefolder_vid_ii,
                                  basename+'_translation-register-params.mat'), 
                     {'disps': disps, 
                      'blurred_frames': blurred_frames})
        
        
    
    
    
    
    
    
    