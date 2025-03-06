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
    
    
    # load the CNN model. 
    from SPOT.Detection.unets import att_unet 
    
    """
    We use only the registered brightfield images to detect organoids
    """

    # specify the export folder (we will work with this. )
    # masterimgfolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/21-SPOT-Drug screen-dye/Caspase_or_cytox_dye/Videos_Resize_Register'
    # masterimgfolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/21-SPOT-Drug screen-dye/Caspase_or_cytox_dye_redo/Videos_Resize_Register'
    
    # # masterimgfolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/26-SPOT-Drug screen-dye/Caspase or Cytotox Only/Videos_Resize_Register'
    # # masterimgfolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/27-SPOT-Drug screen-dye/Caspase or Cytotox Only/Videos_Resize_Register'
    
    # masterimgfolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/30-SPOT-Drug screen-dye/Caspase or Cytotox Only/Videos_Resize_Register'

    # masterimgfolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/26-SPOT-Drug screen-dye_invert/Caspase or Cytotox Only/Videos_Resize_Register'
    # masterimgfolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/27-SPOT-Drug screen-dye_invert/Caspase or Cytotox Only/Videos_Resize_Register'
    
    masterimgfolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/30-SPOT-Drug screen-dye_invert/Caspase or Cytotox Only/Videos_Resize_Register'


    # 
    imgfolders = os.listdir(masterimgfolder)
    imgfolders = np.hstack([os.path.join(masterimgfolder,
                                          ff) for ff in imgfolders if os.path.isdir(os.path.join(masterimgfolder,
                                                                                                 ff))])

     
    # =============================================================================
    #   Load the segmentation model  
    # =============================================================================
    pred_size = 64
    
    segment_model_weights_file = os.path.join('/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/Analysis_Scripts_w_Library/Models/segment_CNN_model',
                                              'organoid-bbox_attn_seg-unet-master-v2_64x64-8feats_v2.h5')
                                      
    segment_model = att_unet(img_w=pred_size, 
                                img_h=pred_size, 
                                n_label=1, 
                                img_ch=3,
                                depth=4, 
                                features=8, 
                                data_format='channels_last')
    segment_model.load_weights(segment_model_weights_file)
    
    print(segment_model.summary())
    
    
    # =============================================================================
    #     Set parameters for segmentation. 
    # =============================================================================
    segment_thresh = .6 # was 0.6 
    # segment_thresh = .5
    # segment_thresh = .25
    # segment_min_I = 0 
    segment_min_I = 0.1 
    
    invert_intensity = True # this was False! consequently we might not have captured many dark organoids.  ### not as good a segmentation!. 
    # invert_intensity = False
    # """
    # Set post filters for the tracking of the organoids. 
    # """
    """
    Load bounding box tracks and set a filter for the tracked bounding boxes, keeping only those that are most stable i.e. have long lifetimes relative to when they appeared 
    """
    track_stability_thresh = 0.25
    ds = 1.
    filter_prob_thresh = 0.05 # higher for confocal => this seems to miss some APC?
    filter_life_thresh =.1 # do we need anything here? => too stringent? 
    
    
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
        """
        load the tracked bbox file 
        """
        bboxfile = os.path.join(savefolder_vid_ii, 'flow_bbox_tracks_Ch-%d.mat' %(desired_ch+1))
        bbox_tracks_obj = spio.loadmat(bboxfile)
        
        
        
        
        imfile = os.path.join(imgfolders[vid_ii],
                                  basename+'_register.tif')
        
        vid = skio.imread(imfile)
        
        if len(vid.shape) == 4:
            vid = vid[...,0]


        
        """
        filter tracked organoids 
        """
        bbox_tracks = bbox_tracks_obj['tracks']
        bbox_tracks_prob = bbox_tracks_obj['prob_tracks']
        bbox_tracks_lifetime = bbox_tracks_obj['track_lifetime_ratio'].ravel()
        
        print(bbox_tracks.shape)
        
        
        filter_prob_index = np.nanmean(bbox_tracks_prob, axis=1) >= filter_prob_thresh
        filter_lifetime_index = bbox_tracks_lifetime >= filter_life_thresh
        keep_track_index = np.logical_and(filter_prob_index, filter_lifetime_index)       
        
        print('filtered tracks with, probfilter: ', filter_prob_thresh, ' lifetime: ', filter_life_thresh)
            
            
        """
        do segmentation and save out. 
        """
        # specify the savefolder for the segmentation 
        savefolder_vid_ii_segmentation = os.path.join(masterimgfolder, 
                                                      basename,
                                                      'organoid_segmentation'); 
        fio.mkdir(savefolder_vid_ii_segmentation)
        
        
        # create folder to save some visualizations. 
        saveplotsfolder = os.path.join(savefolder_vid_ii_segmentation,
                                       'Channel-'+str(desired_ch+1));
        savemoviefolder = os.path.join(saveplotsfolder, 'CNN_boundaries')
        fio.mkdir(savemoviefolder) # this will also create the higher level folder
        
        
        
        if np.sum(keep_track_index) > 0:
        # we have tracks to segment
        
            keep_bbox_tracks = bbox_tracks[keep_track_index].copy()     
            
            
            # enforce that all predicted bbox is valid.
            bbox_val_size = np.logical_and(keep_bbox_tracks[...,2]>keep_bbox_tracks[...,0]+1, # add this in...! 
                                           keep_bbox_tracks[...,3]>keep_bbox_tracks[...,1]+1)
            keep_bbox_tracks[np.logical_not(bbox_val_size>0),:] = np.nan # those that are invalid should be np.nan
            
            
            # run this for all timepoints. 
            vid_bbox_tracks_segs, vid_bbox_tracks_contours = SPOT_track.segment_organoid_bbox_track(vid,
                                                                                                     keep_bbox_tracks, # a densified track array 
                                                                                                     segment_model, # bboxes are already loaded in Yolo format.   
                                                                                                     segment_size=(pred_size, pred_size), 
                                                                                                     min_I=segment_min_I,
                                                                                                     segment_thresh=segment_thresh, # what value to extract contours at after the segmentation 
                                                                                                     # ds_factor = ds,
                                                                                                     clip_boundary_border=10, # this is to constrain the segmentation contour to the initial detected bounding box. 
                                                                                                     smooth_boundary_iter = 2, # is this not working? # smooth this. => is this giving trouble? 
                                                                                                     boundary_sample_pts = 200,
                                                                                                     scale_method='scale',
                                                                                                     invert_intensity=invert_intensity)#,
                                                                                                     # data_format='channels_last') # 100 doesn't quite capture fully complex geometries. # another way would be considering fourier coefficients. 
    
            plotcolors = sns.color_palette('Spectral', len(vid_bbox_tracks_contours)) # set up some plotting colors
            
            
            for frame_no in np.arange(len(vid)):
    
                fig, ax = plt.subplots(figsize=(10,10)) # slightly higher resolution. 
                plt.title('Frame: %d' %(frame_no+1))
                vid_overlay = vid[frame_no,...,:].copy() # reverse to abide by the fact we doing BGR here. 
    
                ax.imshow(vid_overlay, cmap='gray')
                # ax.imshow(vid_overlay)
                contours_tt = vid_bbox_tracks_contours[:, frame_no].copy()
    
                for cont_ii in np.arange(len(contours_tt)):
                    contour_tt_ii = contours_tt[cont_ii]
                    if ~np.isnan(contour_tt_ii[0,0]):
                                ax.plot(contour_tt_ii[:,1], 
                                        contour_tt_ii[:,0], '-', lw=5, color = plotcolors[cont_ii]) # increase this -> show colors to indicate tracking. 
                ax.set_xlim([0, vid[0].shape[1]-1])
                ax.set_ylim([vid[0].shape[0]-1, 0])
                plt.axis('off')
                plt.grid('off')
                fig.savefig(os.path.join(savemoviefolder, 
                                         'Frame-%s.png' %(str(frame_no).zfill(3))), bbox_inches='tight')
                plt.show()
                plt.close(fig)
        
        else:
            # exit. # there are no tracks 
            vid_bbox_tracks_segs = []
            vid_bbox_tracks_contours = []
            keep_bbox_tracks = [] # empty
    
    
        """
        Save the segmentations
        """
        savematfile = os.path.join(savefolder_vid_ii_segmentation,
                                   'org_boundaries-'+'Channel-'+str(desired_ch+1)+'.mat')
    

        # save the dominant statistics.
        spio.savemat(savematfile, {'boundaries': vid_bbox_tracks_contours, 
                                   'boundaries_segs': vid_bbox_tracks_segs, 
                                   'cnn_model_path': segment_model_weights_file,
                                   'pred_size': pred_size,
                                   'keep_track_index': keep_track_index,
                                   'keep_bbox_tracks': keep_bbox_tracks,
                                   'bboxfile': bboxfile,
                                   'filter_prob_thresh': filter_prob_thresh,
                                   'filter_life_thresh': filter_life_thresh, 
                                   'segment_min_I': segment_min_I, 
                                   'segment_thresh': segment_thresh})
            
            
            
            
            
            