# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 11:00:02 2020

@author: felix
"""


def compile_boundaries_arrays(expt, boundaries, total_vid_frames):
    
    import numpy as np 
   
    filenames_all = []
    conditions_all = []
    genetics_all = []
    img_channel_no_all = []
    org_id_all = []
    frame_no_all = []
    pixel_resolution_all = []
    frame_duration_all = []
    total_vid_frames_all = []
    
    boundaries_all_export = [] 
    
    for channel_ii, boundaries_channel in enumerate(boundaries):
        # -> iterate over channels.
#        print(boundaries_channel.shape)
        
        for bb_ii, boundaries_channel_ii in enumerate(boundaries_channel):
            # -> iterate over timepoints. 
#                print(boundaries_channel_ii.shape)
            # bb_ii is the org_id. 
            for bb_ii_tt, boundaries_ii_tt in enumerate(boundaries_channel_ii):
#                    print(len(metrics_names))
                # if features are nan do nothing
#                    print(boundaries_ii_tt.shape) # this can also be empty!.
                # these are now coordinates.... 
#                    if len(boundaries_ii_tt) > 0:
                if ~np.isnan(boundaries_ii_tt[0][0]):
                    
                    filenames_all.append(expt)
                    img_channel_no_all.append(channel_ii+1)
                    org_id_all.append(bb_ii+1)
                    frame_no_all.append(bb_ii_tt+1)
                    pixel_resolution_all.append(pixel_res)
                    frame_duration_all.append(time_res)
                    total_vid_frames_all.append(total_vid_frames) # this is causing the issue
                    
#                            print(boundaries_ii_tt.shape)
                    boundaries_all_export.append(boundaries_ii_tt)
                        
    
#    print(np.array(boundaries_all_export).shape)
#    print(filenames_all)
    filenames_all = np.hstack(filenames_all)
    # conditions_all = np.hstack(conditions_all)
    # genetics_all = np.hstack(genetics_all)
    img_channel_no_all = np.hstack(img_channel_no_all)
    org_id_all = np.hstack(org_id_all)
    frame_no_all = np.hstack(frame_no_all)
    pixel_resolution_all = np.hstack(pixel_resolution_all)
    frame_duration_all = np.hstack(frame_duration_all)
    total_vid_frames_all = np.hstack(total_vid_frames_all)
#    
##    print(filenames_all.shape)
##    print(len(boundaries_all_export))
#    # the creation of the array kills? 
    boundaries_all_export = np.array(boundaries_all_export)
#    print(boundaries_all_export.shape)
    
    return (filenames_all, 
            img_channel_no_all, org_id_all, frame_no_all, 
            pixel_resolution_all, frame_duration_all, total_vid_frames) , boundaries_all_export



def construct_metrics_table_csv(expt, metrics, metricslabels, total_vid_frames,
                                pixel_res=1, 
                                time_res=1):
    
    import pandas as pd 
    import numpy as np 

    all_data = []
    metrics_names = [name.strip() for name in metricslabels]
   
    for channel_ii, metrics_channel in enumerate(metrics):
            
        # do nothing otherwise.
        for bb_ii, metrics_ii in enumerate(metrics_channel):
            
            # bb_ii is the org_id. 
            for bb_ii_tt, metrics_ii_tt in enumerate(metrics_ii):
#                    print(len(metrics_names))
                # if features are nan do nothing
                
                if ~np.isnan(metrics_ii_tt[0]):
                    data = np.hstack([ expt, # filename
                                       channel_ii+1, #want the image channel.
                                       bb_ii+1,  # want the oranoid id.
                                       bb_ii_tt+1, # want the frame_no.
                                       pixel_res,
                                       time_res, 
                                       total_vid_frames, 
                                       metrics_ii_tt, 
                                        ])
#                        print(len(data))
#                        print(data[:11])
                    all_data.append(data)
                        
    all_data = np.array(all_data)
    headers = np.hstack(['Filename',
                         'Img_Channel_No', 
                         'Org_ID', 
                         'Frame_No', 
                         'pixel_resolution[um]',
                         'Frame_Duration[h]',
                         'Total_Video_Frame_No',
                         metrics_names])
                        
#    print(all_data.shape, headers.shape)
    
    all_data = pd.DataFrame(all_data, 
                            index=None,
                            columns=headers)
    
    return all_data



if __name__=="__main__":
    
    """
    Given the computed SAM phenomes for individual, this example compiles the SAM phenomes for both videos together 
    """

    import numpy as np 
    import os 
    import pylab as plt 
    from tqdm import tqdm 
    import scipy.io as spio 
    import skimage.transform as sktform 
    
    """
    loads for SPOT
    """
    import SPOT.Utility_Functions.file_io as fio
    
    
    """
    Specify pixel and time resolutions
    """
    # this is arbitrary unit i.e. we are just going to use the pixel 
    pixel_res = 1 # um 
    time_res = 1 # h
        
    
    """
    Specify the rootfolder where one has downloaded the single cell tracking challenge video dataset to 
    """
    rootfolder = r'C:\Users\fyz11\Documents\Work\Projects\Lu-Organoids\Paper\Suppl_GitHub_Code\test_cell-tracking-challenge_results\PhC-C2DH-U373_analysis'
    
    """
    Specify the top-level save folder we will save the intermediate outputs to.
    """
    saverootfolder = r'C:\Users\fyz11\Documents\Work\Projects\Lu-Organoids\Paper\Suppl_GitHub_Code\test_cell-tracking-challenge_results\PhC-C2DH-U373_analysis\SPOT'
    
    """
    Specify which of the videos will be analyzed. Here, both of the provided will be.
    """
    cellfolders = [r'PhC-C2DH-U373\01',  
                   r'PhC-C2DH-U373\02'] 


    # favorite cell. --- the ruffling cells. 
    for cellfolder in tqdm(cellfolders[:]): # this is the U373
        
        infolder = os.path.join(rootfolder,cellfolder)
        rootname, basename = os.path.split(cellfolder)
        
        """
        Specify save folder location
        """
        saveresultsfolder = os.path.join(saverootfolder, rootname, basename); # mkdir(saveresultsfolder)
        saveresultsfile = os.path.join(saveresultsfolder, 'seg_boundaries_and_tracks.pickle')
        
        expt = rootname+'_'+basename
        
        """
        Grab the results. 
        """
        results = fio.read_pickle(saveresultsfile)
        
        # grab the contours and raw 
        raw = results['raw'].copy()
        contour_xy_array = results['contour_xy_array_filter'].copy()
        seg = results['seg_masks'].copy()
        
        # grab the div_times as extra meta information! 
        div = results['div_times'] # this is just the dividing times for each cell... # this needs to be turned into a suitable array. 
        
        div_bool = np.zeros((len(div), len(raw)), dtype=bool) # N_cells vs time.
        for nn in np.arange(len(div)):
            if div[nn] > -1: 
                div_bool[nn, div[nn]] = 1
        
        
        """
        We wrote a custom compile here, since we don't have a specific table of metadata and for simplification 
        """
        # make this work with the native
        (filenames, img_channel_no, org_id, frame_no_id, pix_res, frame_duration, total_vid_no), boundary_table = compile_boundaries_arrays(expt, 
                                                                                                                                            [contour_xy_array], 
                                                                                                                                            total_vid_frames=len(raw))
                
        
        spio.savemat(os.path.join(saveresultsfolder, 
                                  'final_boundaries.mat'),  # should be suffixed... 
                     {'filenames': filenames,
                      'org_id': org_id, 
                      'frame_no': frame_no_id, 
                      'pix_res': pix_res, 
                      'frame_duration': frame_duration, 
                      'boundary': boundary_table})
        
        
        """
        Shape
        """
        savemetricfile = os.path.join(saveresultsfolder, expt+'_shape-feats.mat')
        shape_feats_obj = spio.loadmat(savemetricfile)
        
        shape_feats = shape_feats_obj['metrics'].copy()
        shape_metric_names = np.hstack([ss.strip() for ss in np.hstack(shape_feats_obj['metric_names'].ravel())])
        shape_norm_bool = shape_feats_obj['metric_norm_bool'].copy()
        
        
        """
        # note we used a custom compile here for the metrics files, since we don't have a specific table of metadata and for simplification 
        """
        shape_table = construct_metrics_table_csv(expt, 
                                                  [shape_feats], 
                                                  shape_metric_names,
                                                  total_vid_frames=len(raw))
        
        
        # """
        # Construct a div_bool table and add this to the shape -> as good as any lol . 
        # """
        div_table = np.zeros(len(shape_table), dtype=bool)
        # for entry_ii, entry in enumerate(shape_table):
        for entry_ii in np.arange(len(shape_table)):
            entry = shape_table.iloc[entry_ii]
            row_ii = int(entry['Org_ID'])-1
            frame_ii = int(entry['Frame_No'])-1
            
            div_table[entry_ii] = div_bool[row_ii, frame_ii]
        
        shape_table['div_bool'] = div_table
        
        shape_table.to_csv(os.path.join(saveresultsfolder, 
                                  'final_shape_metrics.csv'), index=None)
        

        """
        Appearance
        """
        savemetricfile = os.path.join(saveresultsfolder, expt+'_appearance-feats.mat')
        shape_feats_obj = spio.loadmat(savemetricfile)

        shape_feats = shape_feats_obj['metrics'].copy()
        shape_metric_names = np.hstack([ss.strip() for ss in np.hstack(shape_feats_obj['metric_names'].ravel())])
        shape_norm_bool = shape_feats_obj['metric_norm_bool'].copy()
        
        
        shape_table = construct_metrics_table_csv(expt, 
                                                  [shape_feats], 
                                                  shape_metric_names,
                                                  total_vid_frames=len(raw))
        shape_table.to_csv(os.path.join(saveresultsfolder, 
                                  'final_appearance_metrics.csv'), index=None)
        
        print(shape_table)
        
        """
        Motion
        """
        savemetricfile = os.path.join(saveresultsfolder, expt+'_motion-feats.mat')
        shape_feats_obj = spio.loadmat(savemetricfile)
        # spio.savemat(savemetricfile, {'expt': expt, 
        #                               'analysisfile':saveresultsfile,
        #                               'metric_names': metrics_labels, 
        #                               'metric_norm_bool': metrics_norm_bool, 
        #                               'metrics': metrics})        
        shape_feats = shape_feats_obj['metrics'].copy()
        shape_metric_names = np.hstack([ss.strip() for ss in np.hstack(shape_feats_obj['metric_names'].ravel())])
        shape_norm_bool = shape_feats_obj['metric_norm_bool'].copy()
        
        
        shape_table = construct_metrics_table_csv(expt, 
                                                  [shape_feats], 
                                                  shape_metric_names,
                                                  total_vid_frames=len(raw))
        shape_table.to_csv(os.path.join(saveresultsfolder, 
                                  'final_motion_metrics.csv'), index=None)
        
        print(shape_table)
        # =============================================================================
        #      Do the cropping here as well as the division information!. 
        # =============================================================================
        
        patch_size = (64,64)
        all_patches = []
        all_patches_rescale = [] 
        patches_all_size_original = []
        
        for row_ii in np.arange(len(boundary_table))[:]:
            
            boundary_ii = boundary_table[row_ii,:].copy()
            frame_no_ii = frame_no_id[row_ii] - 1
        
            # make the crop patches. 
            # compute the boundary
            bbox = [int(np.min(boundary_ii[:,1])),
                    int(np.min(boundary_ii[:,0])),
                    int(np.ceil(np.max(boundary_ii[:,1])))+1,
                    int(np.ceil(np.max(boundary_ii[:,0])))+1]
            
            
            # #check the patch
            # plt.figure()
            # plt.imshow(raw[frame_no_ii])
            # plt.plot(boundary_ii[:,1],
            #           boundary_ii[:,0],
            #           'k-')
            # plt.show()            
        
            patch = raw[frame_no_ii, bbox[1]:bbox[3], bbox[0]:bbox[2]].copy() # cut this out. 
            all_patches.append(patch)
            patches_all_size_original.append(np.hstack(patch.shape[:2]))
            
            
            # if the patch size is invalid.... 
            if patch.shape[0] == 0 and patch.shape[1]>0:
                patch = np.zeros((1, patch.shape[1]))
            if patch.shape[1] ==0 and patch.shape[0]>0:
                patch = np.zeros((patch.shape[0], 1))
                
                
            # plt.figure()
            # plt.imshow(patch)
            # plt.show()
                
            patch = np.uint8(sktform.resize(patch, patch_size, preserve_range=True))
            all_patches_rescale.append(patch)
            
        all_patches_rescale = np.array(all_patches_rescale)
        # do we rescale this? # prob best not to? 
        savematfile_patches = os.path.join(saveresultsfolder, 
                                           'final_img_patches_noresize.mat')
            
        
        spio.savemat(savematfile_patches, 
                      {'patches_all': all_patches, # use the non-resized version!. 
                       # 'patches_all': patches_all,
                       # 'bbox_sizes_all': bbox_sizes_all, 
                       'patch_size': np.array(patches_all_size_original)}, 
                      do_compression=True)
            
            
        
        
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        