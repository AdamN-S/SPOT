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
    import scipy.io as spioo
    
    import SPOT.Utility_Functions.file_io as fio 
    import SPOT.Tracking.track as SPOT_track
    import SPOT.Tracking.optical_flow as SPOT_optical_flow
    
    import scipy.io as spio 
    
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
                                          ff) for ff in imgfolders])

 
    # """
    # Define settings of the tracker 
    # """
    # # setup some parameters for Farneback optical flow computation 
    optical_flow_params = SPOT_optical_flow.get_default_optical_flow_params()
    
    # setup tracking parameters. 
    iou_match = 0.25 # this is the required iou match across frames.
    # ds_factor = 4
    ds_factor = 1
    min_aspect_ratio = 3 
    max_dense_bbox_cover = 8 # decreasing this to 4 reduces the length able to be tracked .... # should we include an appearance cost? 
    wait_time = 5
    
    
    """
    Set the desired channel 
    """
    desired_ch = 0 # this is default for grayscale (only one channel)
    


    for vid_ii in tqdm(np.arange(len(imgfolders))[:]):
        
        # get the basename. 
        basename = os.path.split(imgfolders[vid_ii])[-1]
        
        # specify the savefolder for this detection part. 
        savefolder_vid_ii = os.path.join(masterimgfolder, 
                                          basename,
                                          'bbox_detection'); 
        # fio.mkdir(savefolder_vid_ii)
        
        
        imfile = os.path.join(imgfolders[vid_ii],
                                  basename+'_register.tif')
        
        vid = skio.imread(imfile)
        
        if len(vid.shape) == 4:
            vid = vid[...,0]


        """
        precompute optical flow
        """
        vid_flow = SPOT_optical_flow.extract_vid_optflow(vid, 
                                                     flow_params=optical_flow_params, 
                                                     rescale_intensity=True)
        vid_flow = np.array(vid_flow).astype(np.float32)
    
        
        """
        Load computed bounding boxes
        """
        bboxfolder = masterimgfolder
        outfolder = savefolder_vid_ii
        
        
        vid_bboxes = fio.fetch_channel_boxes(imfile, bboxfolder, 
                                             ch_no=desired_ch, 
                                             ending='_register.tif',
                                             subfolder='bbox_detection')
        
        print(vid_bboxes)
        print('============')
        
        # we now read and sort the boxes into chronological order
        vid_bboxes = fio.read_detected_bboxes_from_file_for_tracking(vid_bboxes)
    
    
        vid_bbox_tracks_prob, vid_bbox_tracks, vid_bbox_match_checks, (vid_bbox_tracks_all_lens, vid_bbox_tracks_all_start_time, vid_bbox_tracks_lifetime_ratios) = SPOT_track.track_organoid_bbox(vid_flow,
                                                                                                                                                                                                    vid_bboxes, # bboxes are already loaded in yolo format.   
                                                                                                                                                                                                    vid = vid,
                                                                                                                                                                                                    iou_match=iou_match,
                                                                                                                                                                                                    ds_factor = ds_factor,
                                                                                                                                                                                                    wait_time = wait_time,
                                                                                                                                                                                                    min_aspect_ratio=min_aspect_ratio,
                                                                                                                                                                                                    max_dense_bbox_cover=max_dense_bbox_cover,
                                                                                                                                                                                                    to_viz=True,
                                                                                                                                                                                                    saveanalysisfolder=None)
        """
        Save the tracking 
        """
        spio.savemat(os.path.join(outfolder, 'flow_bbox_tracks_Ch-%d.mat' %(desired_ch+1)), 
                      {'tracks':vid_bbox_tracks, 
                      'prob_tracks': vid_bbox_tracks_prob, 
                      'track_quality': vid_bbox_match_checks, 
                      'track_lens': vid_bbox_tracks_all_lens,
                      'track_start_time': vid_bbox_tracks_all_start_time,
                      'track_lifetime_ratio': vid_bbox_tracks_lifetime_ratios})
        
        