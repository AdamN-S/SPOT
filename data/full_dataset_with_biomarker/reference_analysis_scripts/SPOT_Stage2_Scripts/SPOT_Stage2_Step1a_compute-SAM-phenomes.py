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
    import skimage.exposure as skexposure
    from tqdm import tqdm 
    import scipy.io as spio
    import seaborn as sns 
    
    
    import SPOT.Utility_Functions.file_io as fio 
    import SPOT.Tracking.track as SPOT_track
    import SPOT.Tracking.optical_flow as SPOT_optical_flow
    import SPOT.Features.features as SPOT_SAM_features
    
    

    """
    We use only the registered brightfield images to detect organoids
    """

    # specify the export folder (we will work with this. )
    # masterimgfolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/21-SPOT-Drug screen-dye/Caspase_or_cytox_dye/Videos_Resize_Register'
    # masterimgfolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/21-SPOT-Drug screen-dye/Caspase_or_cytox_dye_redo/Videos_Resize_Register'
    
    # masterimgfolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/26-SPOT-Drug screen-dye/Caspase or Cytotox Only/Videos_Resize_Register'
    # masterimgfolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/27-SPOT-Drug screen-dye/Caspase or Cytotox Only/Videos_Resize_Register'
    # masterimgfolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/30-SPOT-Drug screen-dye/Caspase or Cytotox Only/Videos_Resize_Register'

    
    # masterimgfolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/26-SPOT-Drug screen-dye_invert/Caspase or Cytotox Only/Videos_Resize_Register'
    # masterimgfolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/27-SPOT-Drug screen-dye_invert/Caspase or Cytotox Only/Videos_Resize_Register'
    masterimgfolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/30-SPOT-Drug screen-dye_invert/Caspase or Cytotox Only/Videos_Resize_Register'

    

    imgfolders = os.listdir(masterimgfolder)
    # imgfolders = np.hstack([os.path.join(masterimgfolder,
                                          # ff) for ff in imgfolders])    
    imgfolders = np.hstack([os.path.join(masterimgfolder,
                                          ff) for ff in imgfolders if os.path.isdir(os.path.join(masterimgfolder,
                                                                                                 ff))])




    """
    We need to specify the time sampling and spatial resolution 
    """
    # not sure. 
    pixel_res = 1.63 * (2048 / 512.) # um with correction for the downsampling. 
    # time_res = 2 # h # this is weird.
    time_res = 4 # every 4h imaged. 
    rev_channels = False # optionally specify if the channels should be reversed. 
    # exptname = '21-SPOT-Drug screen-dye' # set a global name for this set of videos to call experiment. 
    # exptname = '26-SPOT-Drug screen-dye_invert' # set a global name for this set of videos to call experiment. 
    # exptname = '27-SPOT-Drug screen-dye_invert' # set a global name for this set of videos to call experiment. 
    exptname = '30-SPOT-Drug screen-dye_invert'

    ds = 1 # no downsampling 

    for vid_ii in tqdm(np.arange(len(imgfolders))[:]):
        
        # get the basename. 
        basename = os.path.split(imgfolders[vid_ii])[-1]
        
        
        imfile = os.path.join(imgfolders[vid_ii],
                                  basename+'_register.tif')
        
        vid = skio.imread(imfile)
        
        if len(vid.shape) == 4:
            vid = vid[...,0]

        # since grayscale need to fake RGB
        vid = vid[...,None]

        """
        load the segmentation files for all channels now. 
        """
        # specify the savefolder for the segmentation 
        savefolder_vid_ii_segmentation = os.path.join(masterimgfolder, 
                                                      basename,
                                                      'organoid_segmentation'); 
        
        # load the final boundary-based segmentation. 
        savematfile = os.path.join(savefolder_vid_ii_segmentation, basename+'_boundaries_final_RGB.mat')
        # savematfile = os.path.join(savematfolder, basename+'_boundaries_final_RGB.mat') 
        
        # load 
        boundaries_organoids_RGB = spio.loadmat(savematfile)['boundaries_smooth_final']
        
        

        """
        prepare data. 
        """
        imshape = vid.shape[1:-1]
        
        """
        0. Precompute optical flow for each image channel so we can compute motion features more easily. 
        """
        optical_flow_params = SPOT_optical_flow.get_default_optical_flow_params()
        
        
        flow_channels = []
        
        for ch in np.arange(vid.shape[-1]):
            
            img_ch = vid[...,ch].copy()
            img_ch = np.array([skexposure.rescale_intensity(frame) for frame in img_ch])
            flow_ch = SPOT_optical_flow.extract_vid_optflow(img_ch, 
                                                             flow_params=optical_flow_params,
                                                             rescale_intensity=True)
            flow_channels.append(flow_ch)
            
        if rev_channels:
            flow_channels = flow_channels[::-1]
            boundaries_organoids_RGB = boundaries_organoids_RGB[::-1] # reverse. 
        
        
        
        savematfolder = savefolder_vid_ii_segmentation
        # fio.mkdir(savematfolder)
        
        
        """
        1. Compute and save the metrics into separate .mat files. (save as .pkl files using fio.write_pickle in the same way if individual file is > 4GB)
            a) shape features.
            b) image appearance features
            c) motion features.
        """
        
        # a) shape features. 
        all_metrics = []
          
        for boundaries_ii, boundaries in enumerate(boundaries_organoids_RGB):
        
            if len(boundaries) > 0 :
                
                print('computing shape metrics for Channel %d' %(boundaries_ii+1))
                metrics, metrics_labels, metrics_norm_bool = SPOT_SAM_features.compute_boundary_morphology_features(boundaries, 
                                                                                                                      imshape,  
                                                                                                                      all_feats_compute=True, 
                                                                                                                      contour_curve_feats=True, 
                                                                                                                      curve_order=4, 
                                                                                                                      curve_err=1., 
                                                                                                                      geom_features=True,
                                                                                                                      morph_features=True,
                                                                                                                      fourier_features=True, 
                                                                                                                      shape_context_features=True,
                                                                                                                      norm_scm=True,
                                                                                                                      n_ref_pts_scm=5,
                                                                                                                      pixel_xy_res = pixel_res)
            
                all_metrics.append(metrics)
                print(metrics.shape)
            else:
                all_metrics.append([])
            
            
        savemetricfile = os.path.join(savematfolder, basename+'_final_cont_RGB_shape-feats.mat')
        spio.savemat(savemetricfile, {'expt': exptname, 
                                      'imgfile':imfile,
                                      'rgb_rev': rev_channels, 
                                      'metric_names': metrics_labels, 
                                      'metric_norm_bool': metrics_norm_bool, 
                                      'metrics': all_metrics,
                                      'pixel_res': pixel_res})


        # b) Image Appearance (textural) features. 
        all_metrics = []
          
        for boundaries_ii, boundaries in enumerate(boundaries_organoids_RGB):
        
            if len(boundaries) > 0 :
                print('computing motion metrics for Channel %d' %(boundaries_ii+1))
                metrics, metrics_labels, metrics_norm_bool = SPOT_SAM_features.compute_boundary_texture_features(vid[...,boundaries_ii], 
                                                                                                                boundaries, 
                                                                                                                use_gradient_vid=False,
                                                                                                                compute_all_feats =True,
                                                                                                                compute_intensity_feats=True,
                                                                                                                compute_contour_intensity_feats=True,
                                                                                                                n_contours=3,
                                                                                                                n_angles=1,
                                                                                                                angle_start=None,
                                                                                                                compute_sift=True,
                                                                                                                siftshape=(64,64),
                                                                                                                compute_haralick=True,
                                                                                                                haralick_distance=15)
                all_metrics.append(metrics)
            else:
                all_metrics.append([])
                
        savemetricfile = os.path.join(savematfolder, basename+'_final_cont_RGB_image-feats.mat')
        spio.savemat(savemetricfile, {'expt': exptname, 
                                      'imgfile':imfile,
                                      'rgb_rev': rev_channels, 
                                      'metric_names': metrics_labels, 
                                      'metric_norm_bool': metrics_norm_bool, 
                                      'metrics': all_metrics})


        # c) Motion features. 
        all_metrics = []
          
        for boundaries_ii, boundaries in enumerate(boundaries_organoids_RGB):
        
            if len(boundaries) > 0 :
                print('computing motion metrics for Channel %d' %(boundaries_ii+1))
    #                vid_flow = flow_channels[boundaries_ii].copy()
                metrics, metrics_labels, metrics_norm_bool = SPOT_SAM_features.compute_boundary_motion_features(flow_channels[boundaries_ii], 
                                                                                                                  boundaries, 
                                                                                                                  compute_all_feats =True,
                                                                                                                  compute_global_feats=True,
                                                                                                                  compute_contour_feats=True,
                                                                                                                  n_contour_feat_bins=8,
                                                                                                                  cnt_sigma=3., 
                                                                                                                  n_contours=3,
                                                                                                                  n_angles=1,
                                                                                                                  angle_start=None,
                                                                                                                  pixel_res=pixel_res,
                                                                                                                  time_res=time_res,
                                                                                                                  compute_sift_feats=True,
                                                                                                                  siftshape=(64,64))
                
                all_metrics.append(metrics)
            else:
                all_metrics.append([])
                
        savemetricfile = os.path.join(savematfolder, basename+'_final_cont_RGB_motion-feats.mat')
        spio.savemat(savemetricfile, {'expt': exptname, 
                                      'imgfile':imfile,
                                      'rgb_rev': rev_channels, 
                                      'metric_names': metrics_labels, 
                                      'metric_norm_bool': metrics_norm_bool, 
                                      'metrics': all_metrics,
                                      'pixel_res': pixel_res,
                                      'time_res': time_res})
        
        
        
        
        
        
        
        
        