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
    
    masterimgfolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/26-SPOT-Drug screen-dye_invert/Caspase or Cytotox Only/Videos_Resize_Register'
    # masterimgfolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/27-SPOT-Drug screen-dye_invert/Caspase or Cytotox Only/Videos_Resize_Register'
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
    
    # newest
    exptname = '26-SPOT-Drug screen-dye' # set a global name for this set of videos to call experiment. 
    # exptname = '27-SPOT-Drug screen-dye'
    # exptname = '30-SPOT-Drug screen-dye'
# 
    # ds = 1 # no downsampling 

    for vid_ii in tqdm(np.arange(len(imgfolders))[:]):
        
        # get the basename. 
        basename = os.path.split(imgfolders[vid_ii])[-1]
        
        
        imfile = os.path.join(imgfolders[vid_ii],
                                  basename.replace(' BF', '') +'_register.tif')
        
        print(imfile)
        
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
        0. Perform SPOT detection for each channel. 
        """
        # dog_params = 
        
        dots_channels = []
        
        for ch in np.arange(vid.shape[-1]):
            
            img_ch = vid[...,ch].copy()
            img_ch_raw = img_ch.copy()
            img_ch = np.array([skexposure.rescale_intensity(frame) for frame in img_ch])
            
            # do we flat field?
            # flow_ch = SPOT_optical_flow.extract_vid_optflow(img_ch, 
            #                                                   flow_params=optical_flow_params,
            #                                                   rescale_intensity=True)
            # flow_channels.append(flow_ch)
            all_pts = []
            
            for img_frame_ii, img_frame in enumerate(img_ch):
                img_frame_raw = img_ch_raw[img_frame_ii]/255. # so we are still in the range of 0-1. 
                img_frame = img_frame/255.
                img_frame = ndimage.median_filter(img_frame,size=3)
                # # img_frame = np.maximum(img_frame - ndimage.gaussian_filter(img_frame, sigma=5),0)
                # blobs_dog = blob_dog(img_frame, min_sigma=1, max_sigma=5, 
                #                      threshold=0.1,
                #                      overlap=1.0,
                #                      sigma_ratio=1.1)
                # blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2)
            
                # plt.figure(figsize=(10,10))
                # plt.imshow(img_frame,cmap='gray')
                # plt.plot(blobs_dog[:,1],
                #          blobs_dog[:,0],
                #          'r.')
                # plt.show()
                
                # coords = peak_local_max(img_frame-ndimage.gaussian_filter(img_frame,sigma=3),
                #                         min_distance=3,
                #                         threshold_abs=0.025)
                coords = peak_local_max(img_frame-ndimage.gaussian_filter(img_frame,sigma=3),
                                        min_distance=3,
                                        threshold_abs=0.05)
                
                # plt.figure(figsize=(10,10))
                # plt.imshow(img_frame,cmap='gray')
                # plt.plot(coords[:,1],
                #          coords[:,0],
                #          'r.')
                # plt.show()
                import skimage.filters as skfilters 
                coords_raw_threshold = np.mean(img_frame_raw) #+ 1*np.std(img_frame_raw)
                # coords_raw_threshold = skfilters.threshold_otsu(img_frame_raw)
                
                coords_intensity = img_frame_raw[coords[:,1], coords[:,0]].copy()
                coords_keep = coords_intensity>=coords_raw_threshold
                coords_keep = np.ones(len(coords_keep), dtype=bool)
                print('keep_fraction', np.sum(coords_keep)/float(len(coords_keep)))
                
                coords = coords[coords_keep]
                coords_intensity = coords_intensity[coords_keep]
                
                coords_intensity_raw = img_frame_raw[coords[:,1], coords[:,0]].copy()
                
                
                plt.figure(figsize=(10,10))
                plt.imshow(img_frame,cmap='gray')
                plt.plot(coords[:,1],
                         coords[:,0],
                         'r.')
                plt.show()
                
                
                
                pts = np.hstack([coords,
                                 coords_intensity[:,None], 
                                 coords_intensity_raw[:,None]])
                all_pts.append(pts)
        
            dots_channels.append(all_pts)
            
        if rev_channels:
            dots_channels = dots_channels[::-1]
            boundaries_organoids_RGB = boundaries_organoids_RGB[::-1] # reverse. 
        
        img_ch_norm = (img_ch-np.mean(img_ch)) / np.std(img_ch)
        img_ch_norm = img_ch_norm[...,None]
        img_ch_raw = img_ch_raw[...,None]
        
        savematfolder = savefolder_vid_ii_segmentation
        # fio.mkdir(savematfolder)
        
        
        """
        1. Compute and save the metrics into separate .mat files. (save as .pkl files using fio.write_pickle in the same way if individual file is > 4GB)
            a) shape features.
            b) image appearance features
            c) motion features.
        """
        
        # a) additional Image Appearance (intensity + n_spots) features. 
        all_metrics = []
        all_metrics_names = []
          
        for boundaries_ii, boundaries in enumerate(boundaries_organoids_RGB):
        
            if len(boundaries) > 0 :
                
                """
                Custom loop
                """
                print('computing metrics for Channel %d' %(boundaries_ii+1))
                out = [] # placeholder
                out_feat_labels = [] # placeholder
                # out_feat_labels_norm = [] # placeholder

                # if timesample is None:
                timesample = 1

            #    print(scale_factor)
                for b_ii in tqdm(range(len(boundaries))):

                    boundary_ii = boundaries[b_ii]
                    boundary_ii_descriptors = []

                    for tt in range(0,len(boundary_ii),timesample):

                        vid_frame = img_ch_norm[tt, ...,boundaries_ii]
                        vid_frame_raw = img_ch_raw[tt, ...,boundaries_ii]
                        boundary_ii_tt = boundary_ii[tt]

                        if np.isnan(boundary_ii_tt[0][0]) == True:
                            boundary_ii_descriptors.append([np.nan]) # don't compute.
                        else:

                            all_feats = []
                            all_feats_labels = []
                            all_feats_norm_bool = []

                            """
                            Get the binary mask specified by the boundary contours and crop the localised image out.
                            """
                            cnt_binary, bbox = SPOT_SAM_features.contour_to_binary(boundary_ii_tt, shape=None, return_bbox=True)
                            vid_cnt = vid_frame[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1].copy()
                            vid_cnt_raw = vid_frame_raw[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1].copy()
                            
                            spots_frame = dots_channels[boundaries_ii][tt] # get the spots from this frame. 

                            # test for size
                            if vid_cnt.shape[0] ==0 or vid_cnt.shape[1]==0:
                                # then return nan.
                                boundary_ii_descriptors.append([np.nan]) # don't compute
                            else:
                                cnt_binary = sktform.resize(cnt_binary, output_shape=vid_cnt.shape, order=0, preserve_range=True) > 0

                #                print(vid_cnt.shape)
                #                if vid_cnt.shape[0]>haralick_distance and vid_cnt.shape[1] > haralick_distance:
                    #                vid_cnt_im = vid_cnt.copy()
                                # max_dist = np.min(vid_cnt.shape)-1
                                mean_cnt_global = np.nanmean(vid_cnt[cnt_binary>0])

                                all_feats.append(np.hstack([mean_cnt_global]))
                                all_feats_labels.append(np.hstack(['mean_marker_intensity']))
                                
                                
                                mean_cnt_global_raw = np.nanmean(vid_cnt_raw[cnt_binary>0])

                                all_feats.append(np.hstack([mean_cnt_global_raw]))
                                all_feats_labels.append(np.hstack(['mean_marker_intensity_raw']))
                                
                                """
                                Count the number of spots and mean intensity of spots covered by this organoid.
                                """
                                binary_full_frame = np.zeros(vid_frame.shape, dtype=bool)
                                binary_full_frame[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1] = cnt_binary
                                
                                spots_frame_organoid_bool = binary_full_frame[spots_frame[:,0].astype(np.int32),
                                                                              spots_frame[:,1].astype(np.int32)].copy()
                                
                                n_spots = np.sum(spots_frame_organoid_bool)
                                mean_I_spots = np.nanmean(spots_frame[:,2][spots_frame_organoid_bool>0])
                                mean_I_spots_raw = np.nanmean(spots_frame[:,3][spots_frame_organoid_bool>0])
                                
                                all_feats.append(np.hstack([n_spots, mean_I_spots, mean_I_spots_raw]))
                                all_feats_labels.append(np.hstack(['n_spots', 'mean_marker_spot_intensity', 
                                                                   'mean_marker_spot_intensity_raw']))
                                
                                
                                all_feats = np.hstack(all_feats)
                                all_feats_labels = np.hstack(all_feats_labels)
                                # all_feats_norm_bool = np.hstack(all_feats_norm_bool)
    
                                boundary_ii_descriptors.append(all_feats)

                                out_feat_labels = all_feats_labels
                                # out_feat_labels_norm = all_feats_norm_bool
    
                    out.append(boundary_ii_descriptors)
    
                out = np.array(out, dtype=object)

                # make this a regular array for analysis.
                n_org, n_time = boundaries.shape[:2]

                if timesample is not None:
                    n_time = len(np.arange(0, n_time, timesample)) # as we are subsampling...
                n_feat_size = len(out_feat_labels)

                out_array = np.zeros((n_org, n_time, n_feat_size))
                # print(out_array.shape)

                for ii in range(n_org):
                    for jj in range(n_time):
                        val = out[ii,jj]
                        if len(val) == 1:
                            out_array[ii,jj] = np.nan
                        else:
                            out_array[ii,jj] = val
            
                
                all_metrics.append(out_array)
                
            else:
                all_metrics.append([])
                
        metrics_labels = out_feat_labels
                
        
        savemetricfile = os.path.join(savematfolder, basename+'_final_cont_RGB_additional-feats.mat')
        spio.savemat(savemetricfile, {'expt': exptname, 
                                      'imgfile':imfile,
                                      'rgb_rev': rev_channels, 
                                      'metric_names': metrics_labels, 
                                      # 'metric_norm_bool': metrics_norm_bool, 
                                      'metrics': all_metrics})


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
        
        
        
        
        
        
        
        
        