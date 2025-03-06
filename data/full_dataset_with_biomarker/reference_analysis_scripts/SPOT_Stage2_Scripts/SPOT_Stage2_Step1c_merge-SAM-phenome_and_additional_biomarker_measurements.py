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
    
    from skimage.feature import blob_dog, blob_log, blob_doh
    from skimage.feature import peak_local_max
    import scipy.ndimage as ndimage
    
    """
    This script computes additional features from images that are outside of the standard SAM phenome
        this is so we can use the same file structure to analyze later. 
    """
    

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
    masterimgfolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/27-SPOT-Drug screen-dye_invert/Caspase or Cytotox Only/Videos_Resize_Register'
    # masterimgfolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/30-SPOT-Drug screen-dye_invert/Caspase or Cytotox Only/Videos_Resize_Register'

    
    imgfolders = os.listdir(masterimgfolder)
    # imgfolders = np.hstack([os.path.join(masterimgfolder,
                                          # ff) for ff in imgfolders if '.csv' not in ff])
    imgfolders = np.hstack([os.path.join(masterimgfolder,
                                          ff) for ff in imgfolders if os.path.isdir(os.path.join(masterimgfolder,
                                                                                                 ff))])
    print(len(imgfolders))


    """
    We need to specify the time sampling and spatial resolution 
    """
    # not sure. 
    pixel_res = 1.63 * (2048 / 512.) # um with correction for the downsampling. 
    # time_res = 2 # h # this is weird.
    time_res = 4 # every 4h imaged. 
    rev_channels = False # optionally specify if the channels should be reversed. 
    # exptname = '21-SPOT-Drug screen-dye' # set a global name for this set of videos to call experiment. 

    # exptname = '26-SPOT-Drug screen-dye' # set a global name for this set of videos to call experiment. 
    # exptname = '27-SPOT-Drug screen-dye'
    exptname = '30-SPOT-Drug screen-dye'

    # ds = 1 # no downsampling 

    for vid_ii in tqdm(np.arange(len(imgfolders))[:]):
        
        # get the basename. 
        basename = os.path.split(imgfolders[vid_ii])[-1]
        
        
        imfile = os.path.join(imgfolders[vid_ii],
                                  basename.replace(' BF', '') +'_register.tif')
        

        """
        load the segmentation files for all channels now. 
        """
        # specify the savefolder for the segmentation 
        savefolder_vid_ii_segmentation = os.path.join(masterimgfolder, 
                                                      basename,
                                                      'organoid_segmentation'); 
        
        savematfolder = savefolder_vid_ii_segmentation
        # fio.mkdir(savematfolder)
        
        
        save_additional_metricfile = os.path.join(savematfolder, basename+'_final_cont_RGB_additional-feats.mat')
        # spio.savemat(savemetricfile, {'expt': exptname, 
        #                               'imgfile':imfile,
        #                               'rgb_rev': rev_channels, 
        #                               'metric_names': metrics_labels, 
        #                               # 'metric_norm_bool': metrics_norm_bool, 
        #                               'metrics': all_metrics})
        additional_obj = spio.loadmat(save_additional_metricfile)
        
        additional_metrics = additional_obj['metrics']
        additional_metrics_names = additional_obj['metric_names']
        
        save_appear_metricfile = os.path.join(savematfolder, basename+'_final_cont_RGB_image-feats.mat')
        appear_obj = spio.loadmat(save_appear_metricfile)

        appear_metrics = appear_obj['metrics']
        appear_metrics_names = appear_obj['metric_names']


        """
        Combine to receive
        """
        appear_obj['metrics'] = [np.concatenate([appear_metrics[ii],
                                                 additional_metrics[ii]], axis=-1) for ii in np.arange(len(appear_metrics))]
        appear_obj['metric_names'] = np.hstack([appear_metrics_names, 
                                                additional_metrics_names])
        appear_obj['metric_norm_bool'] = np.hstack([appear_obj['metric_norm_bool'], 
                                                   np.hstack([0,0,0])[None,:]])
        
        """
        resave as additional appended
        """
        resave_savemetricfile = os.path.join(savematfolder, basename+'_final_cont_RGB_augment-image-feats.mat')
        spio.savemat(resave_savemetricfile, appear_obj)
        

    #     # c) Motion features. 
    #     all_metrics = []
          
    #     for boundaries_ii, boundaries in enumerate(boundaries_organoids_RGB):
        
    #         if len(boundaries) > 0 :
    #             print('computing motion metrics for Channel %d' %(boundaries_ii+1))
    # #                vid_flow = flow_channels[boundaries_ii].copy()
    #             metrics, metrics_labels, metrics_norm_bool = SPOT_SAM_features.compute_boundary_motion_features(flow_channels[boundaries_ii], 
    #                                                                                                               boundaries, 
    #                                                                                                               compute_all_feats =True,
    #                                                                                                               compute_global_feats=True,
    #                                                                                                               compute_contour_feats=True,
    #                                                                                                               n_contour_feat_bins=8,
    #                                                                                                               cnt_sigma=3., 
    #                                                                                                               n_contours=3,
    #                                                                                                               n_angles=1,
    #                                                                                                               angle_start=None,
    #                                                                                                               pixel_res=pixel_res,
    #                                                                                                               time_res=time_res,
    #                                                                                                               compute_sift_feats=True,
    #                                                                                                               siftshape=(64,64))
                
    #             all_metrics.append(metrics)
    #         else:
    #             all_metrics.append([])
                
    #     savemetricfile = os.path.join(savematfolder, basename+'_final_cont_RGB_motion-feats.mat')
    #     spio.savemat(savemetricfile, {'expt': exptname, 
    #                                   'imgfile':imfile,
    #                                   'rgb_rev': rev_channels, 
    #                                   'metric_names': metrics_labels, 
    #                                   'metric_norm_bool': metrics_norm_bool, 
    #                                   'metrics': all_metrics,
    #                                   'pixel_res': pixel_res,
    #                                   'time_res': time_res})
        
        
        
        
        
        
        
        
        